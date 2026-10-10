from __future__ import annotations

import asyncio
import copy
import logging
import math
import statistics
from argparse import Namespace
from dataclasses import dataclass
from typing import Any

from slime.utils.async_utils import run
from slime.utils.types import Sample

logger = logging.getLogger(__name__)


MC_VALUE_LABEL_SOURCE = "mc_mean"


@dataclass
class MCValueStats:
    mean: float
    std: float
    n: int
    ci_low: float
    ci_high: float
    converged: bool
    rounds: int
    stop_reason: str
    attempted_samples: int = 0
    failed_samples: int = 0


@dataclass
class MCAnchorResult:
    sample_index: int
    position: int
    ratio: float | None
    stats: MCValueStats


def is_mc_value_label_enabled(args: Namespace) -> bool:
    return (
        getattr(args, "critic_label_source",
                "returns") == MC_VALUE_LABEL_SOURCE
        or float(getattr(args, "critic_mc_supervise_coef", 0.0)) > 0.0
    )


def _optional_timeout_seconds(args: Namespace, name: str) -> float | None:
    value = getattr(args, name, None)
    if value is None:
        return None
    value = float(value)
    if value <= 0:
        return None
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite when set, got {value}")
    return value


def _exception_name(exc: BaseException) -> str:
    return type(exc).__name__


def _reward_to_float(args: Namespace, reward: Any) -> float:
    if isinstance(reward, dict):
        reward_key = getattr(args, "reward_key", None)
        if reward_key and reward_key in reward:
            return float(reward[reward_key])
        for key in ("score", "reward", "value"):
            if key in reward:
                return float(reward[key])
        raise ValueError(
            f"Cannot convert reward dict to scalar; keys={list(reward.keys())}")
    return float(reward)


def _compute_stats(
    values: list[float],
    z_value: float,
    converged: bool,
    rounds: int,
    stop_reason: str,
    *,
    attempted_samples: int | None = None,
    failed_samples: int = 0,
) -> MCValueStats:
    n = len(values)
    attempted_samples = n + failed_samples if attempted_samples is None else attempted_samples
    attempted_samples = max(int(attempted_samples), n)
    failed_samples = max(int(failed_samples), 0)
    if n == 0:
        return MCValueStats(
            mean=0.0,
            std=0.0,
            n=0,
            ci_low=0.0,
            ci_high=0.0,
            converged=False,
            rounds=rounds,
            stop_reason=stop_reason,
            attempted_samples=attempted_samples,
            failed_samples=failed_samples,
        )

    mean = float(sum(values) / n)
    std = float(statistics.pstdev(values)) if n > 1 else 0.0
    half = z_value * std / math.sqrt(n) if n > 1 else float("inf")
    return MCValueStats(
        mean=mean,
        std=std,
        n=n,
        ci_low=mean - half if math.isfinite(half) else mean,
        ci_high=mean + half if math.isfinite(half) else mean,
        converged=converged,
        rounds=rounds,
        stop_reason=stop_reason,
        attempted_samples=attempted_samples,
        failed_samples=failed_samples,
    )


def _resolve_mc_ratios(args: Namespace) -> list[float]:
    ratios = getattr(args, "critic_mc_token_ratios", None)
    if ratios is None:
        ratios = getattr(args, "critic_token_ratios", None)
    if ratios is None:
        return []
    if isinstance(ratios, (int, float)):
        ratios = [float(ratios)]
    else:
        ratios = [float(ratio) for ratio in ratios]
    for ratio in ratios:
        if not 0.0 <= ratio <= 1.0:
            raise ValueError(
                f"critic MC token ratios must be in [0, 1], got {ratio}")
    return ratios


def _select_anchor_positions(args: Namespace, sample: Sample) -> list[tuple[int, float | None]]:
    response_len = int(sample.response_length)
    if response_len <= 0 or sample.remove_sample:
        return []

    loss_mask = sample.loss_mask if sample.loss_mask is not None else [
        1] * response_len
    if len(loss_mask) != response_len:
        raise ValueError(
            f"loss mask length {len(loss_mask)} != response length {response_len}")
    valid_positions = [idx for idx, value in enumerate(
        loss_mask) if int(value) != 0]
    if not valid_positions:
        return []

    ratios = _resolve_mc_ratios(args)
    if not ratios:
        raise ValueError(
            "Online MC value labels require --critic-mc-token-ratios or --critic-token-ratios "
            "to keep MC sampling sparse."
        )

    last_valid_idx = len(valid_positions) - 1
    anchors: list[tuple[int, float | None]] = []
    seen: set[int] = set()
    for ratio in ratios:
        ratio_idx = int(round(last_valid_idx * ratio))
        ratio_idx = max(0, min(last_valid_idx, ratio_idx))
        position = int(valid_positions[ratio_idx])
        if position in seen:
            continue
        seen.add(position)
        anchors.append((position, ratio))
    return anchors


def _build_prefix_sample(args: Namespace, sample: Sample, value_position: int) -> Sample:
    from slime.rollout.sglang_rollout import GenerateState

    state = GenerateState(args)
    prefix = copy.deepcopy(sample)
    response_len = int(sample.response_length)
    prompt_len = len(sample.tokens) - response_len
    if prompt_len < 0:
        raise ValueError(
            f"sample token length {len(sample.tokens)} is shorter than response length {response_len}")

    # Training values are response-aligned with logits[prompt_len - 1 + i],
    # i.e. V[i] is the state after exactly i response tokens have been seen.
    # Therefore the MC rollout prefix for label[i] must contain i response
    # tokens, not i + 1.
    prefix_response_len = int(value_position)
    if prefix_response_len < 0 or prefix_response_len >= response_len:
        raise ValueError(
            f"value_position must be in [0, {response_len - 1}], got {value_position}")
    prefix.tokens = list(sample.tokens[: prompt_len + prefix_response_len])
    response_token_ids = sample.tokens[prompt_len: prompt_len +
                                       prefix_response_len]
    prefix.response = state.tokenizer.decode(
        response_token_ids, skip_special_tokens=False)
    prefix.response_length = prefix_response_len
    prefix.reward = None
    prefix.status = Sample.Status.PENDING
    prefix.loss_mask = None
    prefix.rollout_log_probs = None
    prefix.rollout_routed_experts = None
    prefix.teacher_log_probs = None
    prefix.train_metadata = None
    if isinstance(prefix.metadata, dict):
        prefix.metadata = dict(prefix.metadata)
        for key in [
            "mc_value_anchors",
            "mc_value_labels",
            "mc_value_label_masks",
            "mc_value_std_mean",
            "mc_value_n_mean",
            "mc_value_converged_mean",
            "mc_value_rounds_mean",
            "mc_value_attempted_samples_mean",
            "mc_value_failed_samples_mean",
        ]:
            prefix.metadata.pop(key, None)
        prefix.metadata["_mc_value_force_input_ids"] = True
    return prefix


async def _sample_scores(args: Namespace, prefix_sample: Sample, num_samples: int) -> tuple[list[float], int]:
    if num_samples <= 0:
        return [], 0

    from slime.rollout.sglang_rollout import GenerateState, generate_and_rm

    state = GenerateState(args)
    samples = [copy.deepcopy(prefix_sample) for _ in range(num_samples)]
    sample_timeout = _optional_timeout_seconds(
        args, "critic_mc_sample_timeout_seconds")

    async def _generate_one(sample: Sample) -> Sample | BaseException:
        try:
            coro = generate_and_rm(
                args, sample, state.sampling_params.copy(), evaluation=False)
            if sample_timeout is not None:
                return await asyncio.wait_for(coro, timeout=sample_timeout)
            return await coro
        except asyncio.TimeoutError as exc:
            sample.status = Sample.Status.FAILED
            return exc
        except Exception as exc:
            sample.status = Sample.Status.FAILED
            return exc

    tasks = [
        asyncio.create_task(_generate_one(sample))
        for sample in samples
    ]
    generated_results = await asyncio.gather(*tasks, return_exceptions=True)

    generated: list[Sample] = []
    failed_samples = 0
    failure_counts: dict[str, int] = {}
    for result in generated_results:
        if isinstance(result, BaseException):
            failed_samples += 1
            failure_counts[_exception_name(result)] = failure_counts.get(
                _exception_name(result), 0) + 1
            continue
        generated.append(result)

    if getattr(args, "group_rm", False):
        from slime.rollout.rm_hub import batched_async_rm

        rm_failure_reasons: dict[int, str] = {}
        samples_need_reward = [
            sample
            for sample in generated
            if sample.reward is None and sample.status not in {Sample.Status.ABORTED, Sample.Status.FAILED}
        ]
        if samples_need_reward:
            rm_timeout = _optional_timeout_seconds(
                args, "critic_mc_rm_timeout_seconds")
            try:
                reward_coro = batched_async_rm(args, samples_need_reward)
                if rm_timeout is not None:
                    rewards = await asyncio.wait_for(reward_coro, timeout=rm_timeout)
                else:
                    rewards = await reward_coro
                for sample, reward in zip(samples_need_reward, rewards, strict=False):
                    sample.reward = reward
            except asyncio.TimeoutError as exc:
                for sample in samples_need_reward:
                    sample.status = Sample.Status.FAILED
                    rm_failure_reasons[id(sample)] = "rm_timeout"
                logger.warning(
                    "Skip MC reward batch after timeout: samples=%d timeout=%.1fs",
                    len(samples_need_reward),
                    rm_timeout if rm_timeout is not None else -1.0,
                )
            except Exception as exc:
                reason = f"rm_{_exception_name(exc)}"
                for sample in samples_need_reward:
                    sample.status = Sample.Status.FAILED
                    rm_failure_reasons[id(sample)] = reason
                logger.warning(
                    "Skip MC reward batch after error: samples=%d error=%s",
                    len(samples_need_reward),
                    repr(exc),
                )
    else:
        rm_failure_reasons = {}

    scores: list[float] = []
    for sample in generated:
        if sample.reward is None:
            failed_samples += 1
            reason = rm_failure_reasons.get(id(sample), "missing_reward")
            failure_counts[reason] = failure_counts.get(reason, 0) + 1
            continue
        try:
            scores.append(_reward_to_float(args, sample.reward))
        except Exception as exc:
            failed_samples += 1
            failure_counts[f"reward_{_exception_name(exc)}"] = (
                failure_counts.get(f"reward_{_exception_name(exc)}", 0) + 1
            )

    if failed_samples:
        logger.warning(
            "Skipped MC continuation samples: failed=%d/%d reasons=%s",
            failed_samples,
            num_samples,
            failure_counts,
        )
    return scores, failed_samples


async def _run_mc_for_prefix(args: Namespace, prefix_sample: Sample) -> MCValueStats:
    n_min = int(getattr(args, "critic_mc_min_samples", 4))
    n_max = int(getattr(args, "critic_mc_max_samples", 24))
    batch_size = int(getattr(args, "critic_mc_batch_size", 4))
    ci_target = float(getattr(args, "critic_mc_ci_halfwidth_target", 0.10))
    z_value = float(getattr(args, "critic_mc_z_value", 1.96))
    if n_min <= 0 or n_max <= 0 or batch_size <= 0:
        raise ValueError("critic MC sample counts must be positive")
    if n_min > n_max:
        raise ValueError(
            f"critic_mc_min_samples ({n_min}) cannot exceed critic_mc_max_samples ({n_max})")
    if ci_target < 0:
        raise ValueError(
            f"critic_mc_ci_halfwidth_target must be non-negative, got {ci_target}")

    scores: list[float] = []
    rounds = 0
    attempted = 0
    failed_samples = 0
    target_n = n_min
    while True:
        need = max(0, min(batch_size, target_n -
                   len(scores), n_max - attempted))
        if need > 0:
            rounds += 1
            attempted += need
            batch_scores, batch_failed = await _sample_scores(args, prefix_sample, need)
            scores.extend(batch_scores)
            failed_samples += batch_failed

        probe = _compute_stats(
            scores,
            z_value,
            converged=False,
            rounds=rounds,
            stop_reason="running",
            attempted_samples=attempted,
            failed_samples=failed_samples,
        )
        half_width = max(0.0, probe.ci_high - probe.mean)
        if probe.n >= n_min and half_width <= ci_target:
            return _compute_stats(
                scores,
                z_value,
                converged=True,
                rounds=rounds,
                stop_reason="ci",
                attempted_samples=attempted,
                failed_samples=failed_samples,
            )
        if probe.n >= n_max or attempted >= n_max:
            return _compute_stats(
                scores,
                z_value,
                converged=False,
                rounds=rounds,
                stop_reason="max_samples",
                attempted_samples=attempted,
                failed_samples=failed_samples,
            )
        target_n = min(n_max, max(
            target_n + batch_size, len(scores) + batch_size))


async def _run_mc_anchor_job(
    args: Namespace,
    semaphore: asyncio.Semaphore,
    *,
    sample_index: int,
    sample: Sample,
    position: int,
    ratio: float | None,
) -> MCAnchorResult:
    async with semaphore:
        z_value = float(getattr(args, "critic_mc_z_value", 1.96))
        try:
            prefix_sample = _build_prefix_sample(args, sample, position)
            anchor_timeout = _optional_timeout_seconds(
                args, "critic_mc_anchor_timeout_seconds")
            coro = _run_mc_for_prefix(args, prefix_sample)
            if anchor_timeout is not None:
                stats = await asyncio.wait_for(coro, timeout=anchor_timeout)
            else:
                stats = await coro
        except asyncio.TimeoutError:
            stats = _compute_stats([], z_value, False, 0, "anchor_timeout")
            logger.warning(
                "Skip MC anchor after timeout: sample_index=%s position=%s timeout=%.1fs",
                sample_index,
                position,
                float(getattr(args, "critic_mc_anchor_timeout_seconds", -1.0)),
            )
        except Exception as exc:
            stats = _compute_stats([], z_value, False, 0,
                                   f"anchor_error:{_exception_name(exc)}")
            logger.warning(
                "Skip MC anchor after error: sample_index=%s position=%s error=%s",
                sample_index,
                position,
                repr(exc),
            )
    return MCAnchorResult(sample_index=sample_index, position=position, ratio=ratio, stats=stats)


def _anchor_detail(position: int, ratio: float | None, stats: MCValueStats) -> dict[str, Any]:
    return {
        "position": int(position),
        "value_position": int(position),
        "prefix_response_len": int(position),
        "ratio": float(ratio) if ratio is not None else None,
        "mean": float(stats.mean),
        "std": float(stats.std),
        "n": int(stats.n),
        "ci_low": float(stats.ci_low),
        "ci_high": float(stats.ci_high),
        "converged": bool(stats.converged),
        "rounds": int(stats.rounds),
        "stop_reason": stats.stop_reason,
        "attempted_samples": int(stats.attempted_samples),
        "failed_samples": int(stats.failed_samples),
    }


async def _apply_mc_value_labels_async(args: Namespace, samples: list[Sample]) -> dict[str, float]:
    anchor_concurrency = int(getattr(args, "critic_mc_anchor_concurrency", 1))
    anchor_log_interval = int(
        getattr(args, "critic_mc_anchor_log_interval", 0))
    if anchor_concurrency <= 0:
        raise ValueError(
            f"critic_mc_anchor_concurrency must be positive, got {anchor_concurrency}")
    if anchor_log_interval < 0:
        raise ValueError(
            f"critic_mc_anchor_log_interval must be non-negative, got {anchor_log_interval}")

    stats_all: list[MCValueStats] = []
    labels_by_sample: list[list[float]] = []
    masks_by_sample: list[list[int]] = []
    stds_by_sample: list[list[float]] = []
    ns_by_sample: list[list[float]] = []
    details_by_sample: list[list[dict[str, Any]]] = []
    anchor_jobs: list[tuple[int, Sample, int, float | None]] = []

    for sample_index, sample in enumerate(samples):
        labels_by_sample.append([0.0] * int(sample.response_length))
        masks_by_sample.append([0] * int(sample.response_length))
        stds_by_sample.append([0.0] * int(sample.response_length))
        ns_by_sample.append([0.0] * int(sample.response_length))
        details_by_sample.append([])

        for position, ratio in _select_anchor_positions(args, sample):
            anchor_jobs.append((sample_index, sample, int(position), ratio))

    total_anchors = len(anchor_jobs)
    if total_anchors:
        logger.info(
            "Start online MC value labels: anchors=%d, anchor_concurrency=%d, min_samples=%d, "
            "max_samples=%d, batch_size=%d, sample_timeout=%s, rm_timeout=%s, anchor_timeout=%s",
            total_anchors,
            anchor_concurrency,
            int(getattr(args, "critic_mc_min_samples", 4)),
            int(getattr(args, "critic_mc_max_samples", 24)),
            int(getattr(args, "critic_mc_batch_size", 4)),
            _optional_timeout_seconds(
                args, "critic_mc_sample_timeout_seconds"),
            _optional_timeout_seconds(args, "critic_mc_rm_timeout_seconds"),
            _optional_timeout_seconds(
                args, "critic_mc_anchor_timeout_seconds"),
        )
        semaphore = asyncio.Semaphore(anchor_concurrency)
        tasks = [
            asyncio.create_task(
                _run_mc_anchor_job(
                    args,
                    semaphore,
                    sample_index=sample_index,
                    sample=sample,
                    position=position,
                    ratio=ratio,
                )
            )
            for sample_index, sample, position, ratio in anchor_jobs
        ]

        completed = 0
        for task in asyncio.as_completed(tasks):
            result = await task
            completed += 1
            stats_all.append(result.stats)
            labels_by_sample[result.sample_index][result.position] = result.stats.mean
            masks_by_sample[result.sample_index][result.position] = 1 if result.stats.n > 0 else 0
            stds_by_sample[result.sample_index][result.position] = result.stats.std
            ns_by_sample[result.sample_index][result.position] = float(
                result.stats.n)
            details_by_sample[result.sample_index].append(
                _anchor_detail(result.position, result.ratio, result.stats))
            if anchor_log_interval and (completed % anchor_log_interval == 0 or completed == total_anchors):
                logger.info(
                    "Finished online MC value labels: %d/%d anchors", completed, total_anchors)

    for sample, labels, masks, stds, ns, anchor_details in zip(
        samples,
        labels_by_sample,
        masks_by_sample,
        stds_by_sample,
        ns_by_sample,
        details_by_sample,
        strict=False,
    ):
        anchor_details.sort(key=lambda item: item["position"])
        if sample.metadata is None:
            sample.metadata = {}
        sample.metadata["mc_value_labels"] = labels
        sample.metadata["mc_value_label_masks"] = masks
        sample.metadata["mc_value_stds"] = stds
        sample.metadata["mc_value_ns"] = ns
        sample.metadata["mc_value_anchors"] = anchor_details
        if anchor_details:
            sample.metadata["mc_value_std_mean"] = float(
                sum(item["std"] for item in anchor_details) / len(anchor_details))
            sample.metadata["mc_value_n_mean"] = float(
                sum(item["n"] for item in anchor_details) / len(anchor_details))
            sample.metadata["mc_value_converged_mean"] = float(
                sum(1.0 for item in anchor_details if item["converged"]) / len(
                    anchor_details)
            )
            sample.metadata["mc_value_rounds_mean"] = float(
                sum(item["rounds"]
                    for item in anchor_details) / len(anchor_details)
            )
            sample.metadata["mc_value_attempted_samples_mean"] = float(
                sum(item["attempted_samples"]
                    for item in anchor_details) / len(anchor_details)
            )
            sample.metadata["mc_value_failed_samples_mean"] = float(
                sum(item["failed_samples"]
                    for item in anchor_details) / len(anchor_details)
            )
        else:
            sample.metadata["mc_value_std_mean"] = 0.0
            sample.metadata["mc_value_n_mean"] = 0.0
            sample.metadata["mc_value_converged_mean"] = 0.0
            sample.metadata["mc_value_rounds_mean"] = 0.0
            sample.metadata["mc_value_attempted_samples_mean"] = 0.0
            sample.metadata["mc_value_failed_samples_mean"] = 0.0

    if not stats_all:
        return {
            "mc_value/anchors": 0.0,
            "mc_value/valid_anchors": 0.0,
            "mc_value/std_mean": 0.0,
            "mc_value/n_mean": 0.0,
            "mc_value/converged_rate": 0.0,
            "mc_value/rounds_mean": 0.0,
            "mc_value/failed_anchor_rate": 0.0,
            "mc_value/timeout_anchor_rate": 0.0,
            "mc_value/attempted_samples": 0.0,
            "mc_value/failed_samples": 0.0,
            "mc_value/sample_failure_rate": 0.0,
        }

    valid_anchors = sum(1 for stats in stats_all if stats.n > 0)
    failed_anchors = len(stats_all) - valid_anchors
    timeout_anchors = sum(
        1 for stats in stats_all if "timeout" in stats.stop_reason)
    attempted_samples = sum(stats.attempted_samples for stats in stats_all)
    failed_samples = sum(stats.failed_samples for stats in stats_all)
    return {
        "mc_value/anchors": float(total_anchors),
        "mc_value/valid_anchors": float(valid_anchors),
        "mc_value/std_mean": float(sum(s.std for s in stats_all) / len(stats_all)),
        "mc_value/n_mean": float(sum(s.n for s in stats_all) / len(stats_all)),
        "mc_value/converged_rate": float(sum(1.0 for s in stats_all if s.converged) / len(stats_all)),
        "mc_value/rounds_mean": float(sum(s.rounds for s in stats_all) / len(stats_all)),
        "mc_value/failed_anchor_rate": float(failed_anchors / len(stats_all)),
        "mc_value/timeout_anchor_rate": float(timeout_anchors / len(stats_all)),
        "mc_value/attempted_samples": float(attempted_samples),
        "mc_value/failed_samples": float(failed_samples),
        "mc_value/sample_failure_rate": float(failed_samples / attempted_samples) if attempted_samples else 0.0,
    }


def apply_mc_value_labels(args: Namespace, samples: list[Sample]) -> dict[str, float]:
    if not is_mc_value_label_enabled(args):
        return {}
    return run(_apply_mc_value_labels_async(args, samples))
