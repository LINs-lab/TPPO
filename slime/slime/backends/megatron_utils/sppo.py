"""SPPO-specific value, advantage, and critic-loss functions.

SPPO treats each prompt/response pair as one contextual-bandit sample:

* the critic predicts one prompt-level success logit;
* the advantage is ``R - sigmoid(V_logit)`` and is broadcast to response tokens;
* the critic is trained with ``BCEWithLogitsLoss`` against the scalar outcome.

The functions in this module intentionally stay separate from the token-level
PPO value/advantage implementation.
"""

from argparse import Namespace
from collections.abc import Callable

import torch
import torch.nn.functional as F
from megatron.core import mpu

from slime.utils.types import RolloutBatch


def get_sppo_values(
    logits: torch.Tensor,
    *,
    args: Namespace,
    unconcat_tokens: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
    sample_indices: list[int] | None = None,
    with_entropy: bool = False,
    non_loss_data: bool = True,
    max_seq_lens: list[int] | None = None,
    **_: object,
) -> tuple[torch.Tensor, dict[str, list[torch.Tensor]]]:
    """Extract one raw prompt-value logit per sample.

    ``get_responses`` aligns the first response logit with the final prompt
    token, so taking ``chunk[:1]`` gives the same prompt-only value used by
    the official SPPO critic while retaining the existing packed forward path.
    """
    # Local imports avoid making the general loss module depend on SPPO and
    # keep this implementation independent from the ordinary PPO value path.
    from .loss import _allgather_cp_redistribute, get_responses

    value_logits = []
    for logits_chunk, _tokens_chunk in get_responses(
        logits,
        args=args,
        unconcat_tokens=unconcat_tokens,
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        max_seq_lens=max_seq_lens,
    ):
        if logits_chunk.ndim != 2 or logits_chunk.size(-1) != 1:
            raise ValueError(
                "SPPO requires a scalar critic head with response logits shaped "
                f"[response_tokens, 1], got {tuple(logits_chunk.shape)}"
            )
        value_logits.append(logits_chunk)

    response_logits = {"values": value_logits}
    cp_size = mpu.get_context_parallel_world_size()
    if getattr(args, "allgather_cp", False):
        _allgather_cp_redistribute(
            response_logits,
            logits=logits,
            args=args,
            total_lengths=total_lengths,
            response_lengths=response_lengths,
            max_seq_lens=max_seq_lens,
        )
    elif cp_size > 1:
        raise NotImplementedError(
            "SPPO prompt-value extraction with context parallelism requires --allgather-cp."
        )

    prompt_logits = [chunk[:1].reshape(-1) for chunk in response_logits["values"]]
    return torch.empty((0,), device=logits.device), {"values": prompt_logits}


def compute_sppo_advantages_and_returns(
    args: Namespace,
    *,
    rewards: list[float],
    values: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
    kl: list[torch.Tensor] | None = None,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Compute SPPO's scalar residual and broadcast it over response tokens."""
    if len(rewards) != len(values) or len(values) != len(loss_masks):
        raise ValueError(
            "SPPO rewards/values/loss_masks must have equal sample counts: "
            f"{len(rewards)}, {len(values)}, {len(loss_masks)}"
        )
    if kl is not None and len(kl) != len(rewards):
        raise ValueError(
            f"SPPO KL/reward length mismatch: {len(kl)} vs {len(rewards)}"
        )

    advantages = []
    returns = []
    with torch.no_grad():
        for idx, (reward, value, loss_mask) in enumerate(
            zip(rewards, values, loss_masks, strict=True)
        ):
            value_logit = value.reshape(-1)
            if value_logit.numel() != 1:
                raise ValueError(
                    "SPPO expects one prompt value logit per sample, got "
                    f"shape {tuple(value.shape)} for sample {idx}"
                )
            outcome = torch.as_tensor(
                reward,
                device=value_logit.device,
                dtype=torch.float32,
            )
            if kl is not None and float(getattr(args, "kl_coef", 0.0)) != 0.0:
                outcome = outcome - float(args.kl_coef) * kl[idx].sum().to(
                    device=outcome.device,
                    dtype=outcome.dtype,
                )

            value_probability = torch.sigmoid(value_logit[0].float())
            scalar_advantage = outcome - value_probability
            mask = loss_mask.to(
                device=scalar_advantage.device,
                dtype=scalar_advantage.dtype,
            )
            advantages.append(scalar_advantage * mask)
            # The critic target is one scalar outcome, not a token-level return.
            returns.append(outcome.reshape(1).to(dtype=value_logit.dtype))

    return advantages, returns


def sppo_value_loss_function(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    _sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Train the prompt-level SPPO critic with BCEWithLogitsLoss."""
    from .loss import get_responses

    response_logits = []
    for logits_chunk, _tokens_chunk in get_responses(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=batch["total_lengths"],
        response_lengths=batch["response_lengths"],
        max_seq_lens=batch.get("max_seq_lens", None),
    ):
        if logits_chunk.ndim != 2 or logits_chunk.size(-1) != 1:
            raise ValueError(
                "SPPO requires a scalar critic head with response logits shaped "
                f"[response_tokens, 1], got {tuple(logits_chunk.shape)}"
            )
        response_logits.append(logits_chunk)

    response_logits_dict = {"values": response_logits}
    from .loss import _allgather_cp_redistribute

    cp_size = mpu.get_context_parallel_world_size()
    if getattr(args, "allgather_cp", False):
        _allgather_cp_redistribute(
            response_logits_dict,
            logits=logits,
            args=args,
            total_lengths=batch["total_lengths"],
            response_lengths=batch["response_lengths"],
            max_seq_lens=batch.get("max_seq_lens", None),
        )
    elif cp_size > 1:
        raise NotImplementedError(
            "SPPO value loss with context parallelism requires --allgather-cp."
        )

    prompt_logits = torch.cat(
        [chunk[:1].reshape(-1) for chunk in response_logits_dict["values"]],
        dim=0,
    )
    targets = torch.cat(
        [target.reshape(-1)[:1] for target in batch["returns"]],
        dim=0,
    ).to(device=prompt_logits.device, dtype=prompt_logits.dtype)
    if prompt_logits.numel() != targets.numel():
        raise ValueError(
            "SPPO critic logits/targets count mismatch: "
            f"{prompt_logits.numel()} vs {targets.numel()}"
        )

    loss_per_sample = F.binary_cross_entropy_with_logits(
        prompt_logits,
        targets.clamp(0.0, 1.0),
        reduction="none",
    )
    loss = loss_per_sample.mean()
    value_probability = torch.sigmoid(prompt_logits)
    value_accuracy = ((value_probability >= 0.5) == (targets >= 0.5)).float().mean()
    return loss, {
        "sppo_value_loss": loss.detach(),
        "sppo_value_mean": value_probability.mean().detach(),
        "sppo_return_mean": targets.mean().detach(),
        "sppo_value_accuracy": value_accuracy.detach(),
    }
