from .math_utils import grade_answer_verl
from .math_utils import extract_answer as extract_boxed_answer
from .math_dapo_utils import compute_score as compute_score_dapo
from .gpqa import compute_gpqa_reward
from .f1 import f1_score
from .deepscaler import get_deepscaler_rule_based_reward
from slime.utils.types import Sample
from slime.utils.misc import load_function
import asyncio
import json
import logging
import random
import re

import aiohttp

logger = logging.getLogger(__name__)


_shared_session: aiohttp.ClientSession | None = None


def _timeout_reward(sample: Sample):
    label = sample.label
    return {
        "score": 0.0,
        "point": 0.0,
        "acc": False,
        "extracted_gt": label if isinstance(label, list) else [label],
        "extracted_pred": [],
        "scored_by": ["timeout"],
    }


def _get_rm_timeout(args) -> float:
    return float(
        getattr(args, "rm_timeout_seconds", getattr(
            args, "rm_timeout", 60.0)) or 0.0
    )


def _get_shared_session() -> aiohttp.ClientSession:
    global _shared_session
    if _shared_session is None or _shared_session.closed:
        connector = aiohttp.TCPConnector(
            limit=64,
            enable_cleanup_closed=True,
        )
        timeout = aiohttp.ClientTimeout(total=120)
        _shared_session = aiohttp.ClientSession(
            connector=connector, timeout=timeout)
    return _shared_session


def _normalize_exact_answer(value):
    text = str(value if value is not None else "").strip()
    if text.startswith("\\boxed{") and text.endswith("}"):
        extracted = extract_boxed_answer(text)
        if extracted is not None:
            text = extracted
    try:
        parsed = json.loads(text)
        return json.dumps(parsed, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except Exception:
        normalized = re.sub(r"\s+", " ", text).strip().lower()
        choice_match = re.fullmatch(r"\(?([a-z])\)?", normalized)
        if choice_match:
            return choice_match.group(1)
        return normalized


def boxed_exact_reward(response, label):
    pred = extract_boxed_answer(str(response)) or ""
    gt = label[0] if isinstance(label, list) and label else label
    pred_norm = _normalize_exact_answer(pred)
    gt_norm = _normalize_exact_answer(gt)
    score = 1.0 if pred_norm == gt_norm else 0.0
    return {
        "score": score,
        "point": score,
        "acc": score == 1.0,
        "extracted_pred": pred,
        "extracted_gt": gt_norm,
        "scored_by": "boxed_exact",
    }


def exact_reward(response, label):
    gt = label[0] if isinstance(label, list) and label else label
    pred_norm = _normalize_exact_answer(response)
    gt_norm = _normalize_exact_answer(gt)
    score = 1.0 if pred_norm == gt_norm else 0.0
    return {
        "score": score,
        "point": score,
        "acc": score == 1.0,
        "extracted_pred": pred_norm,
        "extracted_gt": gt_norm,
        "scored_by": "exact",
    }


async def remote_rm(args, sample: Sample, max_retries: int = 10):
    payload = {
        "prompt": sample.prompt,
        "response": sample.response,
        "label": sample.label,
    }
    session = _get_shared_session()
    for attempt in range(max_retries):
        try:
            async with session.post(args.rm_url, json=payload) as resp:
                resp.raise_for_status()
                return await resp.json()
        except Exception as e:
            if attempt + 1 >= max_retries:
                logger.warning(
                    f"remote_rm failed after {attempt + 1} attempts: {e}")
                raise
            backoff = min(2**attempt, 30) + random.random()
            logger.info(
                f"remote_rm: {type(e).__name__}, retrying in {backoff:.1f}s ({attempt + 1}/{max_retries})")
            await asyncio.sleep(backoff)


async def async_rm(args, sample: Sample, **kwargs):
    if args.custom_rm_path is not None:
        rm_function = load_function(args.custom_rm_path)
        return await rm_function(args, sample, **kwargs)

    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    rm_type = (metadata.get("rm_type") or args.rm_type or "").strip()
    response = sample.response
    label = sample.label
    if rm_type.startswith("boxed_"):
        response = extract_boxed_answer(response) or ""
        rm_type = rm_type[len("boxed_"):]

    # This function is intended for remote or time-consuming reward model evaluation.
    # Implement the actual logic as needed.
    if rm_type == "remote_rm":
        return await remote_rm(args, sample)
    elif rm_type == "p1":
        try:
            from .p1 import compute_score_p1
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "rm_type='p1' requires optional dependencies from slime.rollout.rm_hub.p1 "
                f"that are not installed: {e}"
            ) from e
        return await asyncio.to_thread(
            compute_score_p1,
            response,
            label if isinstance(label, list) else [label],
            points=sample.metadata.get("points", None) if isinstance(sample.metadata.get("points", None), list)
            and all(isinstance(p, float) for p in sample.metadata.get("points", []))
            else None,
            use_xverify=args.train_use_xverify,
            base_url=args.rm_url,
            question=sample.metadata.get("question", None) if isinstance(
                sample.metadata.get("question", None), str) else None,
        )
    elif rm_type == "boxed_exact":
        return boxed_exact_reward(response, label)
    elif rm_type == "exact":
        return exact_reward(response, label)
    elif rm_type in {"critic_pretrain", "critic-pretrain"}:
        from .critic_pretrain import compute_critic_pretrain_reward
        error_power = getattr(args, "critic_pretrain_reward_error_power", 1.0)
        r = float(compute_critic_pretrain_reward(
            response, label, error_power=error_power))
        key = args.reward_key
        if key:
            return {key: r, "score": r}
        return {"score": r}
    elif rm_type == "deepscaler":
        return get_deepscaler_rule_based_reward(response, label)
    elif rm_type == "dapo":
        return compute_score_dapo(response, label)
    elif rm_type == "math":
        return 1 if grade_answer_verl(response, label) else 0
    elif rm_type == "f1":
        return f1_score(response, label)[0]
    elif rm_type == "gpqa":
        return compute_gpqa_reward(response, label, metadata=metadata)
    elif rm_type == "countdown":
        from .countdown import compute_countdown_reward

        return compute_countdown_reward(response, label)
    elif rm_type == "ifbench":
        from .ifbench import compute_ifbench_reward

        return compute_ifbench_reward(response, label, metadata=metadata)
    elif rm_type == "random":
        return random.randint(0, 1)
    elif rm_type:
        raise NotImplementedError(
            f"Rule-based RM for {rm_type} is not implemented.")
    else:
        raise NotImplementedError("Rule-based RM type is not specified.")


async def async_rm_with_timeout(args, sample: Sample, **kwargs):
    timeout = _get_rm_timeout(args)
    if timeout <= 0:
        return await async_rm(args, sample, **kwargs)
    try:
        return await asyncio.wait_for(async_rm(args, sample, **kwargs), timeout=timeout)
    except asyncio.TimeoutError:
        logger.warning(
            "Reward scoring timed out after %.1fs; assigning zero reward. "
            "response_length=%s label=%s",
            timeout,
            getattr(sample, "response_length", None),
            str(getattr(sample, "label", ""))[:100],
        )
        return _timeout_reward(sample)


async def batched_async_rm(
    args,
    samples: list[Sample],
    **kwargs,
) -> list[int | float]:
    if args.custom_rm_path is not None:
        # Ensure the custom reward function is implemented in batch mode
        rm_function = load_function(args.custom_rm_path)
        return await rm_function(args, samples, **kwargs)

    async def run_with_timeout(sample: Sample):
        return await async_rm_with_timeout(args, sample, **kwargs)

    tasks = [run_with_timeout(sample) for sample in samples]
    rewards = await asyncio.gather(*tasks)
    return rewards
