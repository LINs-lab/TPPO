import logging
import math
from argparse import Namespace
from collections.abc import Callable, Iterator, Sequence
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from megatron.core import mpu
from torch.utils.checkpoint import checkpoint

logger = logging.getLogger(__name__)

from slime.utils.distributed_utils import distributed_masked_whiten
from slime.utils.metric_utils import (
    VALUE_EXPLAINED_VARIANCE_STAT_PREFIX,
    compute_masked_explained_variance_stats,
)
from slime.utils.misc import load_function
from slime.utils.ppo_utils import (
    build_critic_value_loss_masks_by_tokens,
    broadcast_policy_values_by_tokens,
    calculate_log_probs_and_entropy,
    compute_approx_kl,
    compute_gspo_kl,
    compute_opsm_mask,
    compute_policy_loss,
    get_advantages_and_returns_batch,
    get_grpo_returns,
    get_reinforce_plus_plus_baseline_advantages,
    get_reinforce_plus_plus_returns,
    get_segmented_value_difference_advantages_batch,
)
from slime.utils.types import RolloutBatch
from slime.utils.vapo import get_ppo_lambdas, normalize_vapo_token_sum, positive_example_masks

from .cp_utils import (
    all_gather_with_cp,
    get_logits_and_tokens_offset_with_cp,
    get_sum_of_sample_mean,
    slice_log_prob_with_cp,
)


def _get_sample_weights_for_loss(args: Namespace, batch: RolloutBatch) -> list[float] | None:
    if not getattr(args, "critic_pretrain_reweight", False):
        return None
    weights = batch.get("sample_weights")
    n = len(batch["response_lengths"])
    if weights is None:
        return [1.0] * n
    if len(weights) != n:
        raise ValueError(
            f"sample_weights length mismatch: {len(weights)} vs {n}")
    return [float(w) for w in weights]


def _is_hl_gauss_value_loss(args: Namespace) -> bool:
    return getattr(args, "critic_value_loss_type", "mse") == "hl_gauss"


def _use_mc_value_labels(args: Namespace) -> bool:
    return getattr(args, "critic_label_source", "returns") == "mc_mean"


def _learn_tppo(args: Namespace) -> bool:
    return bool(getattr(args, "critic_learn_tppo", False))


def _use_sigmoid_bce_value(args: Namespace) -> bool:
    """Sigmoid + BCE critic WITHOUT the TPPO cumulative-sum reparametrization.

    Each response token's raw scalar head output is treated directly as a value
    logit: value = sigmoid(logit), loss = BCE(logit, return). This is distinct from
    the TPPO logit-space BCE path, which anchors and cumsums the deltas first.
    """
    return bool(getattr(args, "use_bce_value_loss", False))


def _sigmoid_bce_value_logit_clip(args: Namespace) -> float:
    return float(getattr(args, "bce_value_logit_clip", 0.0))


def _tppo_logit_clip(args: Namespace) -> float:
    return float(getattr(args, "critic_tppo_logit_clip", 0.0))


def _use_tppo_local_gradient(args: Namespace) -> bool:
    return bool(getattr(args, "critic_tppo_local_gradient", False))


def _use_tppo_head_space(args: Namespace) -> bool:
    return bool(getattr(args, "critic_tppo_head_space", False))


def _use_separate_v0_head(args: Namespace) -> bool:
    return bool(getattr(args, "critic_separate_v0_head", False))


def _use_separate_v0_head_last_token_only(args: Namespace) -> bool:
    return _use_separate_v0_head(args) and bool(
        getattr(args, "critic_separate_v0_head_last_token_only", False)
    )


def _clipped_value_mse(
    values: torch.Tensor,
    old_values: torch.Tensor,
    targets: torch.Tensor,
    value_clip: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    values_clipfrac = torch.abs(values - old_values) > value_clip
    values_clipped = old_values + (values - old_values).clamp(-value_clip, value_clip)
    loss_clipped = (values_clipped - targets) ** 2
    loss_unclipped = (values - targets) ** 2
    return torch.max(loss_clipped, loss_unclipped), values_clipfrac


def _stats_for_tensor_list(prefix: str, tensors: list[torch.Tensor]) -> dict[str, torch.Tensor]:
    values = [tensor.detach().float().flatten()
              for tensor in tensors if tensor.numel() > 0]
    if not values:
        return {}

    device = values[0].device

    def sum_stat(fn: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
        return torch.stack([fn(value) for value in values]).sum()

    return {
        f"{prefix}_sample_mean": sum_stat(lambda value: value.mean()),
        f"{prefix}_sample_std_mean": sum_stat(lambda value: value.std(unbiased=False)),
        f"{prefix}_sample_min_mean": sum_stat(lambda value: value.min()),
        f"{prefix}_sample_max_mean": sum_stat(lambda value: value.max()),
        f"{prefix}_sample_abs_mean": sum_stat(lambda value: value.abs().mean()),
        f"{prefix}_count_per_sample": torch.tensor(float(sum(value.numel() for value in values)), device=device),
    }


def _values_from_response_logits(
    logits_chunk: torch.Tensor,
    args: Namespace,
) -> tuple[
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    if _use_separate_v0_head(args):
        if logits_chunk.size(-1) != 2:
            raise ValueError(
                "separate-v0 critic expects a two-output head with shape "
                f"[response_tokens, 2], got {tuple(logits_chunk.shape)}"
            )
        raw = logits_chunk
        if raw.size(0) == 0:
            empty = raw.new_empty((0,))
            return empty, empty, raw, empty, empty

        # Channel 0 is the first-token v0 logit.  Channel 1 is the delta
        # logit for each later response token, matching the trained
        # SeparateV0HeadLogitBCE checkpoint layout.
        v0_logit = raw[:1, 0]
        deltas = raw[1:, 1] * float(getattr(args, "critic_delta_scale", 1.0))
        # Keep the anchor's forward value in every cumulative value, but let
        # its own first-token target provide the anchor gradient.  Otherwise
        # every later cumulative loss also backpropagates through v0 and can
        # overwhelm the anchor head.
        detached_v0_logit = v0_logit.detach()
        cumulative_value_logits = torch.cat(
            (v0_logit, detached_v0_logit + torch.cumsum(deltas, dim=0))
        )
        if _use_separate_v0_head_last_token_only(args) and raw.size(0) > 1:
            # Keep the same forward values, but isolate the value-loss
            # gradient to the terminal delta output.
            detached_prefix = cumulative_value_logits[:-1].detach()
            terminal_logit = (
                v0_logit.detach()
                + deltas[:-1].detach().sum()
                + deltas[-1]
            )
            value_logits = torch.cat((detached_prefix, terminal_logit.view(1)))
        else:
            value_logits = cumulative_value_logits
        values = torch.sigmoid(value_logits)
        return values, deltas, raw, value_logits, v0_logit

    if logits_chunk.size(-1) != 1:
        raise ValueError(
            "scalar critic expects a one-output head; use "
            "--critic-separate-v0-head for a two-output checkpoint, got "
            f"{tuple(logits_chunk.shape)}"
        )
    raw = logits_chunk.squeeze(-1)
    if _use_sigmoid_bce_value(args):
        # Non-delta sigmoid+BCE: treat each token's head output directly as a value
        # logit. No cumulative sum, no anchor. value = sigmoid(raw); value_logits = raw.
        values = torch.sigmoid(raw)
        return values, None, raw, raw, None
    if not _learn_tppo(args):
        return raw, None, raw, None, None
    if raw.numel() == 0:
        cumulative_values = None if _use_tppo_head_space(args) else raw
        return raw, raw, raw, cumulative_values, None
    deltas = raw[1:] * float(getattr(args, "critic_delta_scale", 1.0))

    anchor = raw[:1]

    if _use_tppo_local_gradient(args):
        # Keep the cumulative forward values, but stop gradients through the
        # prefix so each value loss updates only its current critic output.
        if deltas.numel() == 0:
            cumulative_values = anchor
        else:
            prefix = torch.cat((anchor, deltas[:-1]))
            cumulative_values = torch.cat(
                (anchor, prefix.detach().cumsum(dim=0) + deltas),
            )
    else:
        cumulative_values = torch.cat((anchor, anchor + torch.cumsum(deltas, dim=0)))
    if _use_tppo_head_space(args):
        return cumulative_values, deltas, raw, None, None
    value_logits = cumulative_values
    values = torch.sigmoid(value_logits)
    return values, deltas, raw, value_logits, None


def _in_dense_critic_only_warmup(args: Namespace, batch: RolloutBatch) -> bool:
    if not getattr(args, "dense_critic_only_warmup", False):
        return False
    if getattr(args, "loss_type", None) != "value_loss":
        return False
    rollout_id = batch.get("rollout_id")
    if rollout_id is None:
        return False
    return int(rollout_id) < int(getattr(args, "num_critic_only_steps", 0) or 0)


def _get_value_target_chunks(args: Namespace, batch: RolloutBatch) -> list[torch.Tensor]:
    if _in_dense_critic_only_warmup(args, batch) or not _use_mc_value_labels(args):
        return batch["returns"]

    target_chunks = batch.get("mc_value_labels")
    if target_chunks is None:
        raise ValueError(
            "--critic-label-source=mc_mean requires rollout_data['mc_value_labels']")
    if len(target_chunks) != len(batch["response_lengths"]):
        raise ValueError(
            f"mc_value_labels/response_lengths length mismatch: {len(target_chunks)} vs {len(batch['response_lengths'])}"
        )
    return target_chunks


def _get_value_loss_masks(args: Namespace, batch: RolloutBatch) -> list[torch.Tensor]:
    use_dense_critic_only_warmup = _in_dense_critic_only_warmup(args, batch)
    loss_masks = batch["loss_masks"]
    if getattr(args, "critic_token_loss", False) and not use_dense_critic_only_warmup:
        loss_masks = build_critic_value_loss_masks_by_tokens(
            args,
            batch["unconcat_tokens"],
            loss_masks,
        )

    if _use_mc_value_labels(args) and not use_dense_critic_only_warmup:
        mc_value_label_masks = batch.get("mc_value_label_masks")
        if mc_value_label_masks is None:
            raise ValueError("--critic-label-source=mc_mean requires rollout_data['mc_value_label_masks']")
        if len(mc_value_label_masks) != len(loss_masks):
            raise ValueError(
                "mc_value_label_masks/loss_masks length mismatch: "
                f"{len(mc_value_label_masks)} vs {len(loss_masks)}"
            )
        loss_masks = [
            loss_mask * mc_mask.to(device=loss_mask.device, dtype=loss_mask.dtype)
            for loss_mask, mc_mask in zip(loss_masks, mc_value_label_masks, strict=False)
        ]
    return loss_masks


def _select_last_valid_value_loss_masks(
    loss_masks: list[torch.Tensor],
) -> list[torch.Tensor]:
    """Keep only the last currently supervised response position per sample."""
    selected_masks = []
    for loss_mask in loss_masks:
        selected = torch.zeros_like(loss_mask)
        valid_positions = torch.nonzero(loss_mask.flatten() > 0, as_tuple=False).flatten()
        if valid_positions.numel() > 0:
            selected.view(-1)[valid_positions[-1]] = 1
        selected_masks.append(selected)
    return selected_masks


def _get_local_value_loss_mask(
    args: Namespace,
    batch: RolloutBatch,
    loss_masks: list[torch.Tensor] | None = None,
) -> torch.Tensor:
    if loss_masks is None:
        loss_masks = _get_value_loss_masks(args, batch)
    if mpu.get_context_parallel_world_size() > 1:
        max_seq_lens = batch.get("max_seq_lens")
        loss_masks = [
            slice_log_prob_with_cp(
                loss_mask,
                total_length,
                response_length,
                args.qkv_format,
                max_seq_lens[i] if max_seq_lens is not None else None,
            )
            for i, (loss_mask, total_length, response_length) in enumerate(
                zip(loss_masks, batch["total_lengths"], batch["response_lengths"], strict=False)
            )
        ]
    return torch.cat([loss_mask.flatten() for loss_mask in loss_masks], dim=0)


def _get_hl_gauss_support(
    args: Namespace,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    num_bins = int(getattr(args, "hlg_num_bins", 101))
    v_min = float(getattr(args, "hlg_v_min", 0.0))
    v_max = float(getattr(args, "hlg_v_max", 1.0))
    sigma_ratio = float(getattr(args, "hlg_sigma_ratio", 0.75))

    if num_bins < 2:
        raise ValueError(
            f"HL-Gauss requires --hlg-num-bins >= 2, got {num_bins}")
    if not math.isfinite(v_min) or not math.isfinite(v_max) or v_max <= v_min:
        raise ValueError(f"Invalid HL-Gauss value range: [{v_min}, {v_max}]")
    if not math.isfinite(sigma_ratio) or sigma_ratio <= 0:
        raise ValueError(
            f"HL-Gauss requires positive --hlg-sigma-ratio, got {sigma_ratio}")

    edges = torch.linspace(v_min, v_max, num_bins + 1,
                           device=device, dtype=dtype)
    centers = 0.5 * (edges[:-1] + edges[1:])
    delta = (v_max - v_min) / num_bins
    sigma = sigma_ratio * delta
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError(
            f"Invalid HL-Gauss sigma={sigma}, derived from sigma_ratio={sigma_ratio}, delta={delta}")
    return edges, centers, sigma


def _hl_gauss_expected_from_logits(logits: torch.Tensor, args: Namespace) -> torch.Tensor:
    _, centers, _ = _get_hl_gauss_support(
        args, device=logits.device, dtype=logits.dtype)
    probs = torch.softmax(logits, dim=-1)
    return (probs * centers.view(1, -1)).sum(dim=-1)


def _hl_gauss_expected_from_probs(probs: torch.Tensor, args: Namespace) -> torch.Tensor:
    _, centers, _ = _get_hl_gauss_support(
        args, device=probs.device, dtype=probs.dtype)
    view_shape = [1] * (probs.ndim - 1) + [-1]
    return (probs * centers.view(*view_shape)).sum(dim=-1)


def _hl_gauss_stats_from_logits(logits: torch.Tensor, args: Namespace) -> dict[str, torch.Tensor]:
    _, centers, _ = _get_hl_gauss_support(
        args, device=logits.device, dtype=torch.float32)
    probs = torch.softmax(logits.float(), dim=-1)
    view_shape = [1] * (probs.ndim - 1) + [-1]
    centers = centers.view(*view_shape)
    mean = (probs * centers).sum(dim=-1)
    var = (probs * (centers - mean.unsqueeze(-1)).square()
           ).sum(dim=-1).clamp_min(0.0)
    entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)
    p_success = (probs * (centers > 0.5).to(dtype=probs.dtype)).sum(dim=-1)
    return {
        "mean": mean,
        "std": var.sqrt(),
        "entropy": entropy,
        "p_success": p_success,
    }


def _hl_gauss_debug_distribution_from_logits(logits: torch.Tensor, args: Namespace) -> dict[str, torch.Tensor]:
    max_tokens = int(getattr(args, "hlg_debug_distribution_max_tokens", 0))
    if max_tokens > 0 and logits.size(0) > max_tokens:
        token_positions = torch.linspace(0, logits.size(
            0) - 1, steps=max_tokens, device=logits.device).round().long()
        token_positions = torch.unique_consecutive(token_positions)
        logits = logits.index_select(0, token_positions)
    else:
        token_positions = torch.arange(logits.size(
            0), device=logits.device, dtype=torch.long)

    stats = _hl_gauss_stats_from_logits(logits, args)
    probs = torch.softmax(logits.float(), dim=-1)
    return {
        "token_positions": token_positions.detach().cpu().to(dtype=torch.int32),
        "probs": probs.detach().cpu().to(dtype=torch.float16),
        "mean": stats["mean"].detach().cpu().to(dtype=torch.float32),
        "std": stats["std"].detach().cpu().to(dtype=torch.float32),
        "entropy": stats["entropy"].detach().cpu().to(dtype=torch.float32),
        "p_success": stats["p_success"].detach().cpu().to(dtype=torch.float32),
    }


def _hl_gauss_target_probs_from_values(
    values: torch.Tensor,
    args: Namespace,
    *,
    dtype: torch.dtype,
    sigma: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    t = values.float()
    clipped_mask = (t < float(getattr(args, "hlg_v_min", 0.0))) | (
        t > float(getattr(args, "hlg_v_max", 1.0)))

    edges, _, fixed_sigma = _get_hl_gauss_support(
        args, device=t.device, dtype=t.dtype)
    if sigma is None:
        sigma_values = fixed_sigma
    else:
        sigma_values = sigma.to(device=t.device, dtype=t.dtype)
        if sigma_values.shape != t.shape:
            raise ValueError(
                f"HL-Gauss target sigma shape mismatch: {tuple(sigma_values.shape)} vs {tuple(t.shape)}")
        sigma_values = sigma_values.clamp_min(torch.finfo(t.dtype).eps)
    t = t.clamp(min=float(getattr(args, "hlg_v_min", 0.0)),
                max=float(getattr(args, "hlg_v_max", 1.0)))
    sqrt2 = math.sqrt(2.0)
    if isinstance(sigma_values, torch.Tensor):
        sigma_values = sigma_values.unsqueeze(-1)
    cdf_evals = torch.erf((edges - t.unsqueeze(-1)) / (sqrt2 * sigma_values))
    z = (cdf_evals[..., -1] - cdf_evals[..., 0]).clamp_min(1e-6)
    probs = (cdf_evals[..., 1:] - cdf_evals[..., :-1]) / z.unsqueeze(-1)
    probs = probs.clamp_min(0.0)
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    return probs.to(dtype=dtype), clipped_mask


def _hl_beta_projected_target_probs_from_params(
    alpha: torch.Tensor,
    beta: torch.Tensor,
    args: Namespace,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    if alpha.shape != beta.shape:
        raise ValueError(
            f"HLG Beta target alpha/beta shape mismatch: {tuple(alpha.shape)} vs {tuple(beta.shape)}")
    _, centers, _ = _get_hl_gauss_support(
        args, device=alpha.device, dtype=torch.float32)
    if float(getattr(args, "hlg_v_min", 0.0)) != 0.0 or float(getattr(args, "hlg_v_max", 1.0)) != 1.0:
        raise ValueError(
            "HLG Beta-projected targets require value support [0, 1].")

    eps = torch.finfo(torch.float32).eps
    centers = centers.clamp(min=eps, max=1.0 - eps)
    alpha = alpha.float().clamp_min(eps)
    beta = beta.float().clamp_min(eps)
    log_probs = (alpha.unsqueeze(-1) - 1.0) * centers.log().view(1, -1)
    log_probs = log_probs + (beta.unsqueeze(-1) - 1.0) * \
        (1.0 - centers).log().view(1, -1)
    probs = torch.softmax(log_probs, dim=-1)
    return probs.to(dtype=dtype)


def _expand_sample_scalars_to_token_values(
    sample_values: list[float],
    token_chunks: list[torch.Tensor],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if len(sample_values) != len(token_chunks):
        raise ValueError(
            f"sample/token chunk length mismatch: {len(sample_values)} vs {len(token_chunks)}")
    expanded = [
        torch.full((chunk.numel(),), float(value), device=device, dtype=dtype)
        for value, chunk in zip(sample_values, token_chunks, strict=False)
    ]
    if not expanded:
        return torch.empty((0,), device=device, dtype=dtype)
    return torch.cat(expanded, dim=0)


def _hl_loo_position_beta_target_probs(
    args: Namespace,
    *,
    returns_chunks: list[torch.Tensor],
    returns: torch.Tensor,
    response_lengths: list[int],
    rewards: list[float] | None,
    group_num_correct: list[float] | None,
    group_size: list[float] | None,
    dtype: torch.dtype,
) -> torch.Tensor:
    if rewards is None:
        raise ValueError(
            "--hlg-target-mode=loo_position_beta requires rollout_data['rewards']")
    if group_num_correct is None or group_size is None:
        raise ValueError(
            "--hlg-target-mode=loo_position_beta requires rollout_data['hlg_group_num_correct'] "
            "and rollout_data['hlg_group_size']"
        )

    sample_success = [1.0 if float(
        reward) > 0.5 else 0.0 for reward in rewards]
    success = _expand_sample_scalars_to_token_values(
        sample_success,
        returns_chunks,
        device=returns.device,
        dtype=torch.float32,
    ).clamp(0.0, 1.0)
    group_correct = _expand_sample_scalars_to_token_values(
        group_num_correct,
        returns_chunks,
        device=returns.device,
        dtype=torch.float32,
    ).clamp_min(0.0)
    group_n = _expand_sample_scalars_to_token_values(
        group_size,
        returns_chunks,
        device=returns.device,
        dtype=torch.float32,
    ).clamp_min(1.0)
    progress = _expand_response_progress_to_token_values(
        returns_chunks,
        response_lengths,
        device=returns.device,
        dtype=torch.float32,
    )

    for name, value in (
        ("success", success),
        ("group_correct", group_correct),
        ("group_size", group_n),
        ("progress", progress),
    ):
        if value.shape != returns.shape:
            raise ValueError(
                f"HLG LOO Beta target {name} shape mismatch: {tuple(value.shape)} vs {tuple(returns.shape)}")

    prior_alpha = float(getattr(args, "hlg_beta_target_prior_alpha", 0.5))
    group_mult = float(
        getattr(args, "hlg_beta_target_group_evidence_mult", 1.0))
    response_evidence = float(
        getattr(args, "hlg_beta_target_response_evidence", 1.0))
    gamma = float(getattr(args, "hlg_beta_target_position_gamma", 1.0))

    n_loo = (group_n - 1.0).clamp_min(0.0)
    k_loo = (group_correct - success).clamp(min=0.0)
    k_loo = torch.minimum(k_loo, n_loo)
    failure_loo = (n_loo - k_loo).clamp_min(0.0)
    response_weight = progress.clamp(0.0, 1.0).pow(gamma) * response_evidence
    alpha = prior_alpha + group_mult * k_loo + response_weight * success
    beta = prior_alpha + group_mult * failure_loo + \
        response_weight * (1.0 - success)
    return _hl_beta_projected_target_probs_from_params(alpha, beta, args, dtype=dtype)


def _expand_response_progress_to_token_values(
    token_chunks: list[torch.Tensor],
    response_lengths: list[int],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if len(response_lengths) != len(token_chunks):
        raise ValueError(
            f"response/token chunk length mismatch: {len(response_lengths)} vs {len(token_chunks)}")

    expanded = []
    for response_length, chunk in zip(response_lengths, token_chunks, strict=False):
        num_tokens = chunk.numel()
        if num_tokens == 0:
            expanded.append(torch.empty((0,), device=device, dtype=dtype))
            continue
        if num_tokens == int(response_length):
            denom = max(int(response_length), 1)
        else:
            # Context-parallel chunks may hold only part of the response. Keep
            # shaping well-defined locally; default mode stays unchanged.
            denom = max(num_tokens, 1)
        progress = torch.arange(
            1, num_tokens + 1, device=device, dtype=dtype) / denom
        expanded.append(progress.clamp(0.0, 1.0))

    if not expanded:
        return torch.empty((0,), device=device, dtype=dtype)
    return torch.cat(expanded, dim=0)


def _hl_gauss_target_sigma_from_context(
    args: Namespace,
    *,
    returns_chunks: list[torch.Tensor],
    returns: torch.Tensor,
    response_lengths: list[int],
    group_success_p: torch.Tensor | None,
    mc_value_stds: list[torch.Tensor] | None = None,
    mc_value_ns: list[torch.Tensor] | None = None,
    use_mc_value_labels: bool = True,
) -> torch.Tensor | None:
    sigma_mode = getattr(args, "hlg_target_sigma_mode", "none")
    if sigma_mode == "none":
        return None

    _, _, base_sigma = _get_hl_gauss_support(
        args, device=returns.device, dtype=torch.float32)
    min_scale = float(getattr(args, "hlg_target_sigma_min_scale", 0.5))
    max_scale = float(getattr(args, "hlg_target_sigma_max_scale", 2.0))
    gamma = float(getattr(args, "hlg_target_sigma_position_gamma", 1.0))

    if sigma_mode == "mc_sem":
        if not use_mc_value_labels:
            return None
        if not _use_mc_value_labels(args):
            raise ValueError(
                "--hlg-target-sigma-mode=mc_sem requires --critic-label-source=mc_mean")
        if mc_value_stds is None or mc_value_ns is None:
            raise ValueError(
                "--hlg-target-sigma-mode=mc_sem requires rollout_data['mc_value_stds'] and ['mc_value_ns']")
        if len(mc_value_stds) != len(returns_chunks) or len(mc_value_ns) != len(returns_chunks):
            raise ValueError(
                "mc_value_stds/mc_value_ns length mismatch: "
                f"stds={len(mc_value_stds)} ns={len(mc_value_ns)} returns_chunks={len(returns_chunks)}"
            )
        std = torch.cat([chunk.reshape(-1).to(device=returns.device,
                        dtype=torch.float32) for chunk in mc_value_stds])
        n = torch.cat([chunk.reshape(-1).to(device=returns.device,
                      dtype=torch.float32) for chunk in mc_value_ns])
        if std.shape != returns.shape or n.shape != returns.shape:
            raise ValueError(
                f"MC SEM sigma shape mismatch: std={tuple(std.shape)} n={tuple(n.shape)} returns={tuple(returns.shape)}"
            )
        sem = std.clamp_min(0.0) / n.clamp_min(1.0).sqrt()
        return torch.clamp_min(sem, base_sigma)

    min_sigma = base_sigma * min_scale
    max_sigma = base_sigma * max_scale
    progress = _expand_response_progress_to_token_values(
        returns_chunks,
        response_lengths,
        device=returns.device,
        dtype=torch.float32,
    )
    if progress.shape != returns.shape:
        raise ValueError(
            f"HL-Gauss target progress shape mismatch: {tuple(progress.shape)} vs {tuple(returns.shape)}")

    position_scale = (1.0 - progress).clamp(0.0, 1.0).pow(gamma)
    sigma = min_sigma + (base_sigma - min_sigma) * position_scale

    if sigma_mode in ("position_group", "position_group_conflict"):
        if group_success_p is None:
            raise ValueError(
                f"--hlg-target-sigma-mode={sigma_mode} requires hlg_group_success_p.")
        group_success_p = group_success_p.to(
            device=returns.device, dtype=torch.float32).clamp(0.0, 1.0)
        if group_success_p.shape != returns.shape:
            raise ValueError(
                f"HL-Gauss group_success_p shape mismatch: {tuple(group_success_p.shape)} vs {tuple(returns.shape)}"
            )
        group_coef = float(getattr(args, "hlg_target_sigma_group_coef", 0.5))
        group_uncertainty = (
            group_success_p * (1.0 - group_success_p)).clamp_min(0.0).sqrt()
        sigma = sigma * (1.0 + group_coef * group_uncertainty)

    if sigma_mode == "position_group_conflict":
        conflict_coef = float(
            getattr(args, "hlg_target_sigma_conflict_coef", 0.5))
        conflict = (returns.float().clamp(0.0, 1.0) - group_success_p).abs()
        sigma = sigma * (1.0 + conflict_coef * conflict)
    elif sigma_mode != "position":
        if sigma_mode != "position_group":
            raise ValueError(
                f"Unsupported hlg_target_sigma_mode: {sigma_mode}")

    return sigma.clamp(min=min_sigma, max=max_sigma)


def _validate_hlg_advantage_std(
    *,
    mode: str,
    advantages: list[torch.Tensor],
    value_std: list[torch.Tensor] | None,
) -> list[torch.Tensor]:
    if value_std is None:
        raise ValueError(
            f"--hlg-advantage-mode={mode} requires HLG critic value_std")
    if len(value_std) != len(advantages):
        raise ValueError(
            f"value_std/advantages length mismatch: {len(value_std)} vs {len(advantages)}")

    std_list = []
    for adv, std in zip(advantages, value_std, strict=False):
        if adv.shape != std.shape:
            raise ValueError(
                f"value_std/advantage shape mismatch: {tuple(std.shape)} vs {tuple(adv.shape)}")
        std_list.append(
            std.to(device=adv.device, dtype=adv.dtype).clamp_min(0.0))
    return std_list


def _apply_hlg_uncertainty_weighted_advantages(
    args: Namespace,
    *,
    advantages: list[torch.Tensor],
    value_std: list[torch.Tensor],
) -> list[torch.Tensor]:
    beta = float(getattr(args, "hlg_adv_beta", 0.0))
    if beta < 0:
        raise ValueError(f"hlg_adv_beta must be non-negative, got {beta}")
    if beta == 0 or not advantages:
        return advantages

    std_list = _validate_hlg_advantage_std(
        mode="uncertainty_weighted",
        advantages=advantages,
        value_std=value_std,
    )
    nonempty_std = [std.float().reshape(-1)
                    for std in std_list if std.numel() > 0]
    if not nonempty_std:
        return advantages

    all_std = torch.cat(nonempty_std, dim=0)
    mean_std = all_std.mean().clamp_min(1e-6)
    adjusted_advantages = []
    for adv, std in zip(advantages, std_list, strict=False):
        normalized_std = std.float() / mean_std
        confidence = 1.0 / (1.0 + beta * normalized_std)
        adjusted_advantages.append(
            adv * confidence.to(device=adv.device, dtype=adv.dtype))

    nonempty_adv = [adv.float().abs().reshape(-1)
                    for adv in advantages if adv.numel() > 0]
    nonempty_adjusted = [adv.float().abs().reshape(-1)
                         for adv in adjusted_advantages if adv.numel() > 0]
    if not nonempty_adv or not nonempty_adjusted:
        return adjusted_advantages

    original_abs_mean = torch.cat(nonempty_adv, dim=0).mean()
    adjusted_abs_mean = torch.cat(nonempty_adjusted, dim=0).mean()
    scale = torch.where(
        adjusted_abs_mean > 1e-6,
        original_abs_mean / adjusted_abs_mean.clamp_min(1e-6),
        torch.ones_like(adjusted_abs_mean),
    )
    return [adv * scale.to(device=adv.device, dtype=adv.dtype) for adv in adjusted_advantages]


def get_responses(
    logits: torch.Tensor,
    *,
    args: Namespace,
    unconcat_tokens: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
    max_seq_lens: list[int] | None = None,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Yield response-aligned `(logits_chunk, tokens_chunk)` pairs per sample.

    After squeezing batch dimension and applying temperature scaling, this
    function extracts the logits and tokens corresponding to response segments
    for each sample. When context parallelism is disabled, it slices directly
    from the concatenated sequence. With context parallelism enabled, it
    handles split sequences across ranks.

    Args:
        logits: Model outputs with shape `[1, T, V]` (policy) or `[1, T, 1]`
            (value). Must be float32.
        args: Configuration containing `rollout_temperature` for scaling.
        unconcat_tokens: List of token tensors (prompt+response) per sample.
        total_lengths: Total sequence lengths (prompt+response) per sample.
        response_lengths: Response segment lengths per sample.

    Yields:
        Tuple of `(logits_chunk, tokens_chunk)` where `logits_chunk` is shape
        `[R, V]` (policy) or `[R, 1]` (value) and `tokens_chunk` is shape `[R]`
        (1D int64), both aligned to response tokens for one sample.
    """
    qkv_format = args.qkv_format

    assert logits.dtype == torch.float32, f"{logits.dtype}"
    assert len(logits.shape) == 3, f"{logits.shape}"

    if qkv_format == "thd":
        assert logits.size(0) == 1, f"{logits.shape}"
        logits = logits.squeeze(0)
    else:
        assert max_seq_lens is not None
        logits = logits.view(-1, logits.size(-1))

    if args.rollout_temperature != 1.0:
        logits = logits.div(args.rollout_temperature)

    cp_size = mpu.get_context_parallel_world_size()
    end = 0
    seq_start = 0
    for i, (tokens, total_length, response_length) in enumerate(
        zip(unconcat_tokens, total_lengths, response_lengths, strict=False)
    ):
        max_seq_len = max_seq_lens[i] if max_seq_lens is not None else None

        if cp_size == 1:
            if qkv_format == "bshd":
                end = max_seq_len * i + total_length
                start = end - response_length
            else:
                end += total_length
                start = end - response_length
            logits_chunk = logits[start - 1: end - 1]
            tokens_chunk = tokens[-response_length:]
        elif args.allgather_cp:
            # DSA: global concat then contiguous CP split. Each rank owns logits for
            # global positions [chunk_start, chunk_end).
            logits_local_len = logits.size(0)
            cp_rank = mpu.get_context_parallel_rank()
            chunk_start = cp_rank * logits_local_len
            chunk_end = chunk_start + logits_local_len

            prompt_length = total_length - response_length
            resp_token_start = seq_start + prompt_length
            resp_token_end = seq_start + total_length
            logit_global_start = resp_token_start - 1
            logit_global_end = resp_token_end - 1

            s = max(logit_global_start, chunk_start)
            e = min(logit_global_end, chunk_end)
            if e <= s:
                logits_chunk = logits[0:0]
                tokens_chunk = tokens[0:0]
            else:
                logits_chunk = logits[s - chunk_start: e - chunk_start]
                tokens_chunk = tokens[(s + 1) - seq_start: (e + 1) - seq_start]
            assert logits_chunk.size(0) == tokens_chunk.size(
                0), f"{logits_chunk.size(0)} vs {tokens_chunk.size(0)}"
        else:
            # TODO: this is super ugly... do better abstraction.
            chunk_size, chunks_offset, logits_offset, tokens_offset = get_logits_and_tokens_offset_with_cp(
                total_length, response_length, qkv_format, max_seq_len
            )

            logits_0, logits_1 = logits[end: end +
                                        chunk_size], logits[end + chunk_size: end + 2 * chunk_size]
            end += 2 * chunk_size

            logits_0 = logits_0[logits_offset[0][0] - chunks_offset[0]
                                [0]: logits_offset[0][1] - chunks_offset[0][0]]
            tokens_0 = tokens[tokens_offset[0][0]: tokens_offset[0][1]]

            logits_1 = logits_1[logits_offset[1][0] - chunks_offset[1]
                                [0]: logits_offset[1][1] - chunks_offset[1][0]]
            tokens_1 = tokens[tokens_offset[1][0]: tokens_offset[1][1]]

            assert logits_0.size(0) == tokens_0.size(
                0), f"{logits_0.size(0)} vs {tokens_0.size(0)}"
            assert logits_1.size(0) == tokens_1.size(
                0), f"{logits_1.size(0)} vs {tokens_1.size(0)}"

            logits_chunk = torch.cat([logits_0, logits_1], dim=0)
            tokens_chunk = torch.cat([tokens_0, tokens_1], dim=0)

        seq_start += total_length

        yield logits_chunk, tokens_chunk


def _allgather_cp_redistribute(
    res: dict[str, list[torch.Tensor]],
    *,
    logits: torch.Tensor,
    args: Namespace,
    total_lengths: list[int],
    response_lengths: list[int],
    max_seq_lens: list[int] | None = None,
) -> None:
    """Redistribute response tensors from allgather-CP layout to zigzag ring-attn layout.

    After allgather context parallelism, each rank holds a contiguous chunk of
    the global sequence.  This helper reconstructs per-sample full response
    tensors via a differentiable all-reduce and re-slices them into the zigzag
    CP pattern expected by downstream code.

    The *res* dict is modified **in-place**.

    Args:
        res: Dict mapping metric names to lists of per-sample tensors.
        logits: Model output used only to determine the local sequence length
            (``logits.size(1)``).
        args: Configuration (needs ``qkv_format``).
        total_lengths: Total sequence lengths (prompt + response) per sample.
        response_lengths: Response segment lengths per sample.
        max_seq_lens: Optional padded max sequence lengths per sample.
    """
    cp_group = mpu.get_context_parallel_group()
    cp_rank = mpu.get_context_parallel_rank()

    logits_local_len = logits.size(1)  # logits shape: [1, T_local, ...]
    chunk_start = cp_rank * logits_local_len
    chunk_end = chunk_start + logits_local_len

    for key, values in res.items():
        # Reconstruct full response tensors with each rank's contiguous contribution
        full_resps = []
        seq_start = 0
        for value, total_length, response_length in zip(values, total_lengths, response_lengths, strict=False):
            prompt_length = total_length - response_length
            logit_global_start = seq_start + prompt_length - 1
            logit_global_end = seq_start + total_length - 1

            s = max(logit_global_start, chunk_start)
            e = min(logit_global_end, chunk_end)

            def zeros(length: int) -> torch.Tensor:
                return torch.zeros(
                    [length] + list(value.shape[1:]),
                    dtype=value.dtype,
                    device=value.device,
                    requires_grad=True,
                )

            if e <= s:
                # This rank has no response logits for this sample.
                full_resp = zeros(response_length)
            else:
                resp_start = s - logit_global_start
                resp_end = e - logit_global_start
                left = zeros(resp_start)
                right = zeros(response_length - resp_end)
                full_resp = torch.cat([left, value, right], dim=0)

            assert full_resp.size(
                0) == response_length, f"Expected {response_length}, got {full_resp.size(0)}"
            full_resps.append(full_resp)
            seq_start += total_length

        # Single differentiable all-reduce to gather full response from all CP ranks
        all_cat = torch.cat(full_resps, dim=0)
        all_cat = dist.nn.all_reduce(all_cat, group=cp_group)

        # Re-slice each sample into zigzag CP pattern
        new_values = []
        for idx, (full_resp, total_length, response_length) in enumerate(
            zip(all_cat.split(response_lengths, dim=0),
                total_lengths, response_lengths, strict=False)
        ):
            max_seq_len = max_seq_lens[idx] if max_seq_lens is not None else None
            new_values.append(
                slice_log_prob_with_cp(
                    full_resp, total_length, response_length, args.qkv_format, max_seq_len)
            )

        res[key] = new_values


def get_log_probs_and_entropy(
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
    hlg_debug_distribution_mask: list[bool] | None = None,
) -> dict[str, list[torch.Tensor]]:
    """Compute per-token log-probabilities (and optionally entropy) on responses.

    For each sample, extracts response-aligned logits and tokens, then computes
    log-probabilities via softmax across the tensor-parallel group. Log-probs
    are squeezed from `[R, 1]` to `[R]`. Entropy values are always appended
    (even when `with_entropy=False`), but only included in the result dict
    when requested.

    Args:
        logits: Policy logits with shape `[1, T, V]`.
        args: Configuration (temperature applied in `get_responses`).
        unconcat_tokens: List of token tensors per sample.
        total_lengths: Total sequence lengths per sample.
        response_lengths: Response segment lengths per sample.
        with_entropy: If True, include "entropy" key in result.
        non_loss_data: Unused; kept for API compatibility.

    Returns:
        Dict with key "log_probs" mapping to a list of `[R]` tensors per
        sample. If `with_entropy` is True, also includes "entropy" key with
        a list of `[R]` tensors.
    """
    assert non_loss_data
    log_probs_list = []
    entropy_list = []
    for logits_chunk, tokens_chunk in get_responses(
        logits,
        args=args,
        unconcat_tokens=unconcat_tokens,
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        max_seq_lens=max_seq_lens,
    ):
        log_prob, entropy = calculate_log_probs_and_entropy(
            logits_chunk,
            tokens_chunk,
            mpu.get_tensor_model_parallel_group(),
            with_entropy=with_entropy,
            chunk_size=args.log_probs_chunk_size,
        )

        log_probs_list.append(log_prob.squeeze(-1))
        entropy_list.append(entropy)

    res = {
        "log_probs": log_probs_list,
    }
    if with_entropy:
        res["entropy"] = entropy_list

    # we need to turn the all gather kv into zigzag ring attn kv
    if args.allgather_cp:
        _allgather_cp_redistribute(
            res,
            logits=logits,
            args=args,
            total_lengths=total_lengths,
            response_lengths=response_lengths,
            max_seq_lens=max_seq_lens,
        )

    return torch.empty((0,), device=logits.device), res


def full_model_score_gradient_loss(
    args: Namespace,
    batch: RolloutBatch,
    coefficient_key: str,
    normalization_samples: int,
    mask_key: str,
    logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, list[str] | torch.Tensor]]:
    """Build a normalized score-function objective for full-model backward.

    The returned scalar is the mean, over responses, of the masked token mean
    ``coefficient_t * log pi(a_t | s_t)``. Megatron's pipeline scheduler then
    differentiates it through every trainable actor parameter. No proxy or
    detached hidden state is involved.
    """
    if normalization_samples <= 0:
        raise ValueError("normalization_samples must be positive")
    if mpu.get_context_parallel_world_size() != 1:
        raise RuntimeError("exact full-model gradient analysis currently requires context_parallel_size == 1")

    _, outputs = get_log_probs_and_entropy(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=batch["total_lengths"],
        response_lengths=batch["response_lengths"],
        with_entropy=False,
        max_seq_lens=batch.get("max_seq_lens"),
    )
    coefficients = batch.get(coefficient_key)
    masks = batch.get(mask_key)
    if coefficients is None or masks is None:
        raise KeyError(
            f"full-model gradient batch is missing {coefficient_key!r} or {mask_key!r}"
        )
    objective = logits.sum() * 0.0
    for log_prob, coefficient, mask in zip(
        outputs["log_probs"], coefficients, masks, strict=True
    ):
        coefficient = torch.as_tensor(coefficient, device=log_prob.device, dtype=log_prob.dtype).reshape(-1)
        mask = torch.as_tensor(mask, device=log_prob.device, dtype=log_prob.dtype).reshape(-1)
        if log_prob.numel() != coefficient.numel() or log_prob.numel() != mask.numel():
            raise ValueError(
                "full-model gradient response length mismatch: "
                f"log_prob={log_prob.numel()}, coefficient={coefficient.numel()}, mask={mask.numel()}"
            )
        denominator = mask.sum().clamp_min(1.0)
        objective = objective + (log_prob * coefficient * mask).sum() / denominator
    objective = objective / float(normalization_samples)
    return (
        objective,
        torch.tensor(1, device=logits.device),
        {
            "keys": ["analysis_score_objective"],
            "values": torch.tensor(
                [len(outputs["log_probs"]), float(objective.detach().item())],
                device=logits.device,
            ),
        },
    )


def _vocab_parallel_softmax(vocab_parallel_logits: torch.Tensor, process_group) -> torch.Tensor:
    """Full-vocab softmax on the LOCAL vocab shard, TP-collective (mirrors _VocabParallelEntropy)."""
    logits_max = vocab_parallel_logits.max(dim=-1, keepdim=True).values
    dist.all_reduce(logits_max, op=dist.ReduceOp.MAX, group=process_group)
    normalized_exp = (vocab_parallel_logits - logits_max).exp()
    denom = normalized_exp.sum(dim=-1, keepdim=True)
    dist.all_reduce(denom, group=process_group)
    return normalized_exp / denom


def _get_lm_head_weight(model: Sequence, args: Namespace) -> torch.Tensor | None:
    """Extract lm_head weight from Megatron model for gradient computation."""
    try:
        # For Megatron models, iterate through model chunks to find output_layer
        for model_chunk in model:
            if hasattr(model_chunk, "module"):
                inner = model_chunk.module
            else:
                inner = model_chunk

            # Check for output_layer (typical in Megatron GPT models)
            if hasattr(inner, "output_layer") and inner.output_layer is not None:
                return inner.output_layer.weight

            # Check for lm_head (alternative naming)
            if hasattr(inner, "lm_head") and inner.lm_head is not None:
                return inner.lm_head.weight

            # For models with decoder -> output_layer structure
            if hasattr(inner, "decoder") and hasattr(inner.decoder, "output_layer"):
                if inner.decoder.output_layer is not None:
                    return inner.decoder.output_layer.weight

        logger.warning("Could not find lm_head/output_layer weight in model")
        return None
    except Exception as exc:
        logger.warning(f"Failed to extract lm_head weight: {exc}")
        return None


def _compute_true_gradients_per_response(
    hidden_states: torch.Tensor,
    tokens: torch.Tensor,
    coefficients: dict[str, torch.Tensor],
    loss_mask: torch.Tensor,
    lm_head_weight: torch.Tensor,
    temperature: float = 1.0,
) -> dict[str, float]:
    """Compute true output-layer gradients via torch.autograd.grad.

    Args:
        hidden_states: [seq_len, hidden_dim] final hidden states (will be detached)
        tokens: [seq_len] target token ids
        coefficients: dict with keys like "return", "baseline", "advantage",
                     each is [seq_len] weights for that gradient type
        loss_mask: [seq_len] mask for valid positions
        lm_head_weight: [vocab, hidden] lm_head weight tensor
        temperature: softmax temperature

    Returns:
        dict of gradient metrics: norms, cosines, etc. (scalars only, not vectors)
    """
    with torch.enable_grad():
        # Detach hidden states to avoid backprop through encoder
        # (we only want output-layer gradient)
        h_detached = hidden_states.detach().requires_grad_(False)  # [seq_len, hidden]

        # Ensure lm_head_weight requires grad
        if not lm_head_weight.requires_grad:
            lm_head_weight.requires_grad_(True)

        # Recompute logits: [seq_len, vocab]
        logits_proxy = h_detached @ lm_head_weight.T  # [seq_len, hidden] @ [hidden, vocab]
        if temperature != 1.0:
            logits_proxy = logits_proxy / temperature

        # Compute log probs
        log_probs = torch.nn.functional.log_softmax(logits_proxy, dim=-1)  # [seq_len, vocab]

        # Extract log_prob for target tokens
        token_log_probs = log_probs.gather(dim=-1, index=tokens.unsqueeze(-1)).squeeze(-1)  # [seq_len]

        # Compute gradients for each coefficient type
        gradients = {}
        for name, coef in coefficients.items():
            # Weighted loss: sum of coef_t * log_prob_t * mask_t
            weighted_loss = (token_log_probs * coef * loss_mask).sum()

            # Compute gradient: ∇_W L
            if lm_head_weight.grad is not None:
                lm_head_weight.grad.zero_()

            grad_outputs = torch.autograd.grad(
                outputs=weighted_loss,
                inputs=lm_head_weight,
                retain_graph=True,
                create_graph=False,
            )

            grad_w = grad_outputs[0]  # [vocab, hidden]
            gradients[name] = grad_w.detach().clone()

        # Compute metrics from gradients
        metrics = {}

        # Norms
        for name, grad in gradients.items():
            metrics[f"norm_{name}"] = float(torch.norm(grad).item())

        # Cosine similarities
        if "return" in gradients and "advantage" in gradients:
            g_return = gradients["return"].flatten()
            g_adv = gradients["advantage"].flatten()
            cos = torch.nn.functional.cosine_similarity(
                g_return.unsqueeze(0), g_adv.unsqueeze(0), dim=1
            )
            metrics["cos_return_adv"] = float(cos.item())

        if "return" in gradients and "baseline" in gradients:
            g_return = gradients["return"].flatten()
            g_base = gradients["baseline"].flatten()
            norm_return = torch.norm(g_return)
            norm_base = torch.norm(g_base)
            metrics["baseline_leakage_ratio"] = float((norm_base / (norm_return + 1e-12)).item())

        return metrics


def _compute_full_model_gradient_norm(
    model: Sequence,
    logits: torch.Tensor,
    tokens: torch.Tensor,
    coefficient: torch.Tensor,
    loss_mask: torch.Tensor,
    temperature: float = 1.0,
) -> float:
    """Compute full-model gradient norm via backward pass.

    Does a complete backward pass and computes the L2 norm of gradients across
    all model parameters. Only the norm is returned (not the gradient vectors).

    Args:
        model: Megatron model sequence
        logits: [seq_len, vocab] already-computed logits
        tokens: [seq_len] target token ids
        coefficient: [seq_len] weights (G_t, V_t, or A_t)
        loss_mask: [seq_len] mask for valid positions
        temperature: softmax temperature (already applied to logits, pass 1.0)

    Returns:
        float: L2 norm of the full-model gradient ||∇_θ L||
    """
    with torch.enable_grad():
        # Collect all trainable parameters
        all_params = []
        for model_chunk in model:
            inner = model_chunk.module if hasattr(model_chunk, "module") else model_chunk
            for param in inner.parameters():
                if param.requires_grad:
                    all_params.append(param)

        if not all_params:
            logger.warning("No trainable parameters found")
            return float('nan')

        # Zero out gradients
        for param in all_params:
            if param.grad is not None:
                param.grad.zero_()

        # Ensure logits require grad
        if not logits.requires_grad:
            logits = logits.detach().requires_grad_(True)

        # Compute log probs
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)  # [seq_len, vocab]

        # Extract log_prob for target tokens
        token_log_probs = log_probs.gather(dim=-1, index=tokens.unsqueeze(-1)).squeeze(-1)  # [seq_len]

        # Weighted loss
        weighted_loss = (token_log_probs * coefficient * loss_mask).sum()

        # Backward to compute gradients
        weighted_loss.backward()

        # Compute gradient norm
        grad_norm_sq = 0.0
        for param in all_params:
            if param.grad is not None:
                grad_norm_sq += torch.sum(param.grad ** 2).item()

        return float(grad_norm_sq ** 0.5)


def _compute_grad_metrics_with_autograd(
    logits: torch.Tensor,
    *,
    args: Namespace,
    unconcat_tokens: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
    sample_indices: list[int] | None,
    max_seq_lens: list[int] | None,
    returns: list[torch.Tensor],
    values: list[torch.Tensor],
    advantages: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
) -> dict[str, list[dict[str, float]]]:
    """Compute true output-layer gradient metrics via torch.autograd.grad.

    This is the NEW implementation that computes real gradients using backward passes.
    Returns per-response gradient metrics (scalars) instead of gradient vectors.

    Args:
        logits: Model output logits
        args: Arguments containing model reference (_analysis_model) and config
        unconcat_tokens: List of token tensors per response
        total_lengths: Total sequence lengths
        response_lengths: Response-only lengths
        sample_indices: Optional sample indices
        max_seq_lens: Max sequence lengths (for bshd format)
        returns: G_t for each response
        values: V_t (baseline) for each response
        advantages: A_t = G_t - V_t for each response
        loss_masks: Masks for valid positions

    Returns:
        dict with keys:
        - "gradient_metrics": list of dicts, each containing:
            - norm_return, norm_baseline, norm_advantage
            - cos_return_adv, baseline_leakage_ratio
            - response metadata
    """
    if mpu.get_context_parallel_world_size() != 1:
        raise RuntimeError("gradient metrics computation requires context_parallel_size == 1")

    device = logits.device
    qkv_format = args.qkv_format

    # Get model and lm_head weight
    model = getattr(args, "_analysis_model", None)
    if model is None:
        logger.warning("_analysis_model not found in args, cannot compute true gradients")
        return {"gradient_metrics": []}

    lm_head_weight = _get_lm_head_weight(model, args)
    if lm_head_weight is None:
        logger.warning("Could not extract lm_head weight, cannot compute true gradients")
        return {"gradient_metrics": []}

    # Get hidden states
    buffer = getattr(args, "_analysis_h_buffer", None)
    h_flat = buffer.pop(0) if buffer else None
    if h_flat is None:
        logger.warning("No hidden states captured, cannot compute true gradients")
        return {"gradient_metrics": []}

    if qkv_format == "thd":
        assert logits.size(0) == 1
        lg = logits.squeeze(0)
    else:
        assert max_seq_lens is not None
        lg = logits.reshape(-1, logits.size(-1))

    if int(h_flat.shape[0]) != int(lg.shape[0]):
        logger.warning(
            f"Hidden state seq len {int(h_flat.shape[0])} != logits seq len {int(lg.shape[0])}"
        )
        return {"gradient_metrics": []}

    temperature = float(args.rollout_temperature) if hasattr(args, "rollout_temperature") else 1.0

    # Process each response
    result_metrics = []
    end = 0

    for i, (tokens, total_length, response_length) in enumerate(
        zip(unconcat_tokens, total_lengths, response_lengths, strict=False)
    ):
        total_length = int(total_length)
        response_length = int(response_length)

        if qkv_format == "bshd":
            end_i = int(max_seq_lens[i]) * i + total_length
            start_i = end_i - response_length
        else:
            end += total_length
            end_i = end
            start_i = end_i - response_length

        # Extract data for this response
        h_response = h_flat[start_i - 1: end_i - 1]  # [response_len, hidden]
        tokens_response = torch.as_tensor(tokens, device=device).reshape(-1)[-response_length:]  # [response_len]

        if int(h_response.shape[0]) != response_length:
            logger.warning(f"Response {i}: hidden state length mismatch, skipping")
            continue

        # Get coefficients
        coefficients = {
            "return": returns[i].to(device).float() if i < len(returns) else torch.zeros(response_length, device=device),
            "baseline": values[i].to(device).float() if i < len(values) else torch.zeros(response_length, device=device),
            "advantage": advantages[i].to(device).float() if i < len(advantages) else torch.zeros(response_length, device=device),
        }

        mask = loss_masks[i].to(device).float() if i < len(loss_masks) else torch.ones(response_length, device=device)

        try:
            # Compute output-layer gradients
            metrics = _compute_true_gradients_per_response(
                hidden_states=h_response,
                tokens=tokens_response,
                coefficients=coefficients,
                loss_mask=mask,
                lm_head_weight=lm_head_weight,
                temperature=temperature,
            )

            # Compute full-model gradients (more expensive)
            # Extract logits for this response
            logits_response = lg[start_i - 1: end_i - 1]  # [response_len, vocab]

            for name, coef in coefficients.items():
                try:
                    norm = _compute_full_model_gradient_norm(
                        model=model,
                        logits=logits_response,
                        tokens=tokens_response,
                        coefficient=coef,
                        loss_mask=mask,
                        temperature=1.0,  # temperature already applied to lg
                    )
                    metrics[f"norm_{name}_full_model"] = norm
                except Exception as exc:
                    logger.warning(f"Failed to compute full-model gradient for {name}: {exc}")
                    metrics[f"norm_{name}_full_model"] = float('nan')

            # Compute full-model leakage ratio
            if "norm_return_full_model" in metrics and "norm_baseline_full_model" in metrics:
                norm_return = metrics["norm_return_full_model"]
                norm_base = metrics["norm_baseline_full_model"]
                if norm_return > 0:
                    metrics["baseline_leakage_ratio_full_model"] = norm_base / norm_return

            # Add metadata
            metrics["response_id"] = i
            if sample_indices is not None and i < len(sample_indices):
                metrics["sample_index"] = int(sample_indices[i]) if sample_indices[i] is not None else i

            result_metrics.append(metrics)

        except Exception as exc:
            logger.warning(f"Failed to compute gradient metrics for response {i}: {exc}")
            continue

    # Compute batch-level statistics: variance reduction and SNR
    batch_metrics = {}

    if len(result_metrics) >= 2:  # Need at least 2 samples for variance
        # Extract norms for each gradient type
        norm_keys = [
            ("norm_return", "norm_baseline", "norm_advantage"),  # output-layer
            ("norm_return_full_model", "norm_baseline_full_model", "norm_advantage_full_model"),  # full-model
        ]

        for norm_return_key, norm_baseline_key, norm_adv_key in norm_keys:
            # Check if this gradient type exists
            if not all(norm_return_key in m for m in result_metrics):
                continue

            # Extract norms
            norms_return = [m[norm_return_key] for m in result_metrics if norm_return_key in m]
            norms_baseline = [m[norm_baseline_key] for m in result_metrics if norm_baseline_key in m]
            norms_adv = [m[norm_adv_key] for m in result_metrics if norm_adv_key in m]

            # Filter out NaN values
            norms_return = [x for x in norms_return if not math.isnan(x)]
            norms_baseline = [x for x in norms_baseline if not math.isnan(x)]
            norms_adv = [x for x in norms_adv if not math.isnan(x)]

            if len(norms_return) < 2 or len(norms_adv) < 2:
                continue

            # Compute variance (treating norms as 1D "vectors")
            mean_return = sum(norms_return) / len(norms_return)
            mean_adv = sum(norms_adv) / len(norms_adv)

            var_return = sum((x - mean_return) ** 2 for x in norms_return) / len(norms_return)
            var_adv = sum((x - mean_adv) ** 2 for x in norms_adv) / len(norms_adv)

            # Variance reduction
            variance_reduction = 1.0 - var_adv / (var_return + 1e-12)

            # SNR = mean / std
            std_return = (var_return ** 0.5)
            std_adv = (var_adv ** 0.5)

            snr_return = abs(mean_return) / (std_return + 1e-12)
            snr_adv = abs(mean_adv) / (std_adv + 1e-12)
            snr_gain = snr_adv / (snr_return + 1e-12)

            # Determine suffix based on gradient type
            suffix = "_full_model" if "full_model" in norm_return_key else "_output_layer"

            batch_metrics[f"variance_reduction{suffix}"] = variance_reduction
            batch_metrics[f"snr_gain{suffix}"] = snr_gain
            batch_metrics[f"gradient_variance_return{suffix}"] = var_return
            batch_metrics[f"gradient_variance_adv{suffix}"] = var_adv
            batch_metrics[f"snr_return{suffix}"] = snr_return
            batch_metrics[f"snr_adv{suffix}"] = snr_adv

    return {"gradient_metrics": result_metrics, "batch_metrics": batch_metrics}


def get_log_probs_and_grad_metrics(
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
    hlg_debug_distribution_mask: list[bool] | None = None,
    returns: list[torch.Tensor] | None = None,
    values: list[torch.Tensor] | None = None,
    advantages: list[torch.Tensor] | None = None,
    loss_masks: list[torch.Tensor] | None = None,
) -> tuple[torch.Tensor, dict[str, list[torch.Tensor] | list[dict]]]:
    """``get_log_probs_and_entropy`` plus TRUE gradient metrics via torch.autograd.grad.

    This is the NEW API that computes real output-layer gradients using backward passes.
    Returns gradient metrics (scalars) instead of gradient proxy vectors.

    This legacy per-response diagnostic is separate from the batch-level
    output-layer analysis used by the export path. It requires:
        returns: List of G_t tensors [response_len] for each response
        values: List of V_t tensors [response_len] for each response
        advantages: List of A_t tensors [response_len] for each response
        loss_masks: List of mask tensors [response_len] for each response

    Returns:
        tuple[torch.Tensor, dict]:
        - empty tensor (for API compatibility)
        - dict containing:
            - "log_probs": list of log probability tensors
            - "gradient_metrics": list of dicts with scalar metrics per response
    """
    # First get log probs as usual
    empty, res = get_log_probs_and_entropy(
        logits,
        args=args,
        unconcat_tokens=unconcat_tokens,
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        sample_indices=sample_indices,
        with_entropy=with_entropy,
        non_loss_data=non_loss_data,
        max_seq_lens=max_seq_lens,
        hlg_debug_distribution_mask=hlg_debug_distribution_mask,
    )

    # Compute true gradient metrics if coefficients are provided
    if returns is not None and values is not None and advantages is not None:
        try:
            grad_metrics = _compute_grad_metrics_with_autograd(
                logits,
                args=args,
                unconcat_tokens=unconcat_tokens,
                total_lengths=total_lengths,
                response_lengths=response_lengths,
                sample_indices=sample_indices,
                max_seq_lens=max_seq_lens,
                returns=returns,
                values=values,
                advantages=advantages,
                loss_masks=loss_masks or [torch.ones(int(r)) for r in response_lengths],
            )
            res.update(grad_metrics)
        except Exception as exc:
            logger.warning(f"Failed to compute gradient metrics: {exc}")
            res["gradient_metrics"] = []
    else:
        logger.warning("Returns/values/advantages not provided, skipping gradient metrics")
        res["gradient_metrics"] = []

    return empty, res


def get_values(
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
    hlg_debug_distribution_mask: list[bool] | None = None,
) -> dict[str, list[torch.Tensor]]:
    """Extract per-token value predictions over response tokens.

    For each sample, extracts response-aligned chunks from the value head
    output and squeezes the final dimension from `[R, 1]` to `[R]`.

    Args:
        logits: Value head output with shape `[1, T, 1]`.
        args: Configuration (passed to `get_responses` which uses
            `rollout_temperature` even though values don't need temperature).
        unconcat_tokens: List of token tensors per sample.
        total_lengths: Total sequence lengths per sample.
        response_lengths: Response segment lengths per sample.
        with_entropy: Unused; kept for signature compatibility.
        non_loss_data: Unused; kept for signature compatibility.

    Returns:
        Dict with key "values" mapping to a list of `[R]` value tensors
        per sample.
    """
    value_list = []
    value_std_list = []
    value_entropy_list = []
    value_p_success_list = []
    hlg_value_distribution_debug = []
    delta_list = []
    raw_value_logit_list = []
    value_logit_list = []
    v0_logit_list = []
    v0_value_list = []
    use_hl_gauss = _is_hl_gauss_value_loss(args)
    keep_tppo_deltas = _learn_tppo(args)
    # Sigmoid+BCE (non-delta) also needs the per-token value logits kept for the loss.
    keep_value_logits = keep_tppo_deltas or _use_sigmoid_bce_value(args)
    need_hlg_std = getattr(args, "hlg_advantage_mode", "none") != "none" or bool(
        getattr(args, "hlg_save_debug_stats", False)
    )
    need_hlg_debug_stats = bool(getattr(args, "hlg_save_debug_stats", False))
    need_hlg_debug_distribution = (
        use_hl_gauss
        and int(getattr(args, "hlg_debug_distribution_samples", 0)) > 0
        and hlg_debug_distribution_mask is not None
    )
    if need_hlg_debug_distribution and len(hlg_debug_distribution_mask) != len(response_lengths):
        raise ValueError(
            "hlg_debug_distribution_mask/response_lengths length mismatch: "
            f"{len(hlg_debug_distribution_mask)} vs {len(response_lengths)}"
        )
    capture_hlg_debug_distribution = need_hlg_debug_distribution and mpu.get_tensor_model_parallel_rank() == 0
    hlg_num_bins = int(getattr(args, "hlg_num_bins", 101))
    for sample_idx, (logits_chunk, _) in enumerate(
        get_responses(
            logits,
            args=args,
            unconcat_tokens=unconcat_tokens,
            total_lengths=total_lengths,
            response_lengths=response_lengths,
            max_seq_lens=max_seq_lens,
        )
    ):
        if use_hl_gauss:
            if logits_chunk.size(-1) != hlg_num_bins:
                raise ValueError(
                    f"HL-Gauss expects value head dim={hlg_num_bins}, but got logits chunk shape {tuple(logits_chunk.shape)}"
                )
            if need_hlg_std or need_hlg_debug_stats:
                stats = _hl_gauss_stats_from_logits(logits_chunk, args)
                value_list.append(stats["mean"])
                value_std_list.append(stats["std"])
                if need_hlg_debug_stats:
                    value_entropy_list.append(stats["entropy"])
                    value_p_success_list.append(stats["p_success"])
            else:
                value_list.append(
                    _hl_gauss_expected_from_logits(logits_chunk, args))
            if need_hlg_debug_distribution:
                debug_entry = None
                if capture_hlg_debug_distribution and bool(hlg_debug_distribution_mask[sample_idx]):
                    debug_entry = _hl_gauss_debug_distribution_from_logits(
                        logits_chunk, args)
                    debug_entry["sample_index"] = (
                        int(sample_indices[sample_idx]
                            ) if sample_indices is not None else sample_idx
                    )
                    debug_entry["response_length"] = int(
                        response_lengths[sample_idx])
                    debug_entry["total_length"] = int(
                        total_lengths[sample_idx])
                hlg_value_distribution_debug.append(debug_entry)
        else:
            expected_dim = 2 if _use_separate_v0_head(args) else 1
            if logits_chunk.size(-1) != expected_dim:
                raise ValueError(
                    f"critic value head dimension mismatch: expected {expected_dim}, "
                    f"got {tuple(logits_chunk.shape)}"
                )
            value, delta, raw_value_logits, value_logits, v0_logit = _values_from_response_logits(
                logits_chunk, args)
            value_list.append(value)
            raw_value_logit_list.append(raw_value_logits)
            if keep_tppo_deltas and delta is not None:
                delta_list.append(delta)
            if keep_value_logits and value_logits is not None:
                value_logit_list.append(value_logits)

    res = {
        "values": value_list,
    }
    if raw_value_logit_list:
        res["critic_raw_value_logits"] = raw_value_logit_list
    if delta_list:
        res["critic_tppo_deltas"] = delta_list
    if value_logit_list:
        res["critic_value_logits"] = value_logit_list
    if value_std_list:
        res["value_std"] = value_std_list
    if value_entropy_list:
        res["value_entropy"] = value_entropy_list
    if value_p_success_list:
        res["value_p_success"] = value_p_success_list

    if args.allgather_cp:
        _allgather_cp_redistribute(
            res,
            logits=logits,
            args=args,
            total_lengths=total_lengths,
            response_lengths=response_lengths,
            max_seq_lens=max_seq_lens,
        )

    # Derive v0 from the response-aligned reconstructed logit sequence after
    # CP redistribution. This keeps the logging tensor shape valid for CP.
    if (
        _use_separate_v0_head(args) or _learn_tppo(args)
    ) and res.get("critic_value_logits"):
        v0_logit_list = [value[:1] for value in res["critic_value_logits"]]
        v0_value_list = [torch.sigmoid(value) for value in v0_logit_list]
        res["critic_v0_logits"] = v0_logit_list
        res["critic_v0_values"] = v0_value_list

    if need_hlg_debug_distribution:
        res["hlg_value_distribution_debug"] = hlg_value_distribution_debug

    return torch.empty((0,), device=logits.device), res


def apply_opd_kl_to_advantages(
    args: Namespace,
    rollout_data: RolloutBatch,
    advantages: list[torch.Tensor],
    student_log_probs: list[torch.Tensor] | None,
) -> None:
    """Apply on-policy distillation KL penalty to advantages.

    Computes reverse KL (student_logp - teacher_logp) and adds weighted penalty
    to advantages in-place. This is orthogonal to the base advantage estimator.

    Args:
        args: Configuration containing `use_opd` and `opd_kl_coef`.
        rollout_data: Dict containing "teacher_log_probs".
        advantages: List of advantage tensors to modify in-place.
        student_log_probs: List of student log-probability tensors.

    References:
        https://github.com/thinking-machines-lab/tinker-cookbook/blob/main/tinker_cookbook/distillation/train_on_policy.py
    """

    if student_log_probs is None:
        return

    teacher_log_probs = rollout_data.get("teacher_log_probs")
    if teacher_log_probs is None:
        raise ValueError(
            f"OPD with opd_type='{args.opd_type}' requires teacher_log_probs, but it is missing.")

    device = student_log_probs[0].device
    teacher_log_probs = [t.to(device=device) for t in teacher_log_probs]

    reverse_kls = []
    for i, adv in enumerate(advantages):
        reverse_kl = student_log_probs[i] - teacher_log_probs[i]
        advantages[i] = adv - args.opd_kl_coef * reverse_kl
        reverse_kls.append(reverse_kl)

    # Store reverse KL for logging
    rollout_data["opd_reverse_kl"] = reverse_kls


def compute_advantages_and_returns(args: Namespace, rollout_data: RolloutBatch) -> None:
    """Compute advantages and returns in-place based on `args.advantage_estimator`.

    This function extracts rewards, log-probs, values, and masks from
    `rollout_data`, computes KL divergences, then applies the chosen advantage
    estimator. Supported methods: "grpo", "gspo", "ppo", "reinforce_plus_plus",
    "reinforce_plus_plus_baseline", and "sppo". When `args.normalize_advantages` is
    True, advantages are whitened across the data-parallel group using masked
    statistics.

    Early returns if both `log_probs` and `values` are None (intermediate
    pipeline stages).

    Args:
        args: Configuration specifying estimator type, KL coefficient,
            normalization settings, and other hyperparameters.
        rollout_data: Dict containing input lists ("log_probs", "ref_log_probs",
            "rewards", "values", "response_lengths", "loss_masks",
            "total_lengths"). Modified in-place to add "advantages" and
            "returns" keys, each mapping to lists of tensors per sample.
    """
    log_probs: list[torch.Tensor] = rollout_data.get(
        "rollout_log_probs" if args.use_rollout_logprobs else "log_probs")
    ref_log_probs: list[torch.Tensor] = rollout_data.get("ref_log_probs")
    rewards: list[float] = rollout_data.get("rewards")
    values: None | list[torch.Tensor] = rollout_data.get("values")
    value_std: None | list[torch.Tensor] = rollout_data.get("value_std")
    response_lengths: list[int] = rollout_data.get("response_lengths")
    loss_masks: list[torch.Tensor] = rollout_data.get("loss_masks")
    total_lengths: list[int] = rollout_data.get("total_lengths")
    max_seq_lens: list[int] | None = rollout_data.get("max_seq_lens", None)

    # return when not the last pp stage.
    if not mpu.is_pipeline_last_stage():
        return

    if args.kl_coef == 0 or not log_probs:
        # when kl_coef is 0, we won't compute ref_log_prob
        xs = log_probs if log_probs is not None else values
        kl = [torch.zeros_like(x, dtype=torch.float32,
                               device=x.device) for x in xs]
    else:
        kl = [
            compute_approx_kl(
                log_probs[i],
                ref_log_probs[i],
                kl_loss_type=args.kl_loss_type,
            )
            for i in range(len(log_probs))
        ]

    if args.advantage_estimator in ["grpo", "gspo"]:
        rewards = torch.tensor(
            rewards, dtype=torch.float32, device=kl[0].device)
        returns = get_grpo_returns(rewards, kl)
        # TODO: is the copy necessary?
        advantages = [r for r in returns]

    elif args.advantage_estimator == "sppo":
        from .sppo import compute_sppo_advantages_and_returns

        if values is None:
            raise ValueError("SPPO requires critic values, but rollout_data['values'] is missing.")
        sppo_kl = kl if len(kl) == len(rewards) else None
        advantages, returns = compute_sppo_advantages_and_returns(
            args,
            rewards=rewards,
            values=values,
            loss_masks=loss_masks,
            kl=sppo_kl,
        )

    elif args.advantage_estimator == "ppo":
        if args.policy_token_value_broadcast:
            tokens = rollout_data.get("tokens")
            if values is None:
                raise ValueError(
                    "--policy-token-value-broadcast requires rollout_data['values']; got None while using PPO."
                )
            if tokens is None:
                raise ValueError(
                    "--policy-token-value-broadcast requires rollout_data['tokens']; got None while using PPO."
                )
            values = broadcast_policy_values_by_tokens(
                args=args,
                values_list=values,
                tokens_list=tokens,
                total_lengths=total_lengths,
                response_lengths=response_lengths,
                loss_masks=loss_masks,
            )
            if value_std is not None:
                value_std = broadcast_policy_values_by_tokens(
                    args=args,
                    values_list=value_std,
                    tokens_list=tokens,
                    total_lengths=total_lengths,
                    response_lengths=response_lengths,
                    loss_masks=loss_masks,
                )
                rollout_data["value_std"] = value_std

        old_rewards = rewards
        rewards = []
        kl_coef = -args.kl_coef
        cp_rank = mpu.get_context_parallel_rank()
        for reward, k in zip(old_rewards, kl, strict=False):
            k *= kl_coef
            if cp_rank == 0:
                k[-1] += reward
            rewards.append(k)
        critic_lambd, policy_lambd = get_ppo_lambdas(args, response_lengths)
        advantages, returns = get_advantages_and_returns_batch(
            total_lengths, response_lengths, values, rewards, args.gamma, critic_lambd,
            policy_lambd=policy_lambd,
        )
        actor_segment_ratios = getattr(args, "ppo_actor_segment_ratios", None)
        if actor_segment_ratios is not None:
            # Only replace the actor signal. The critic keeps the ordinary
            # lambda-return above, isolating segmented credit assignment from
            # the critic-training target.
            advantages = get_segmented_value_difference_advantages_batch(
                total_lengths=total_lengths,
                response_lengths=response_lengths,
                values_list=values,
                rewards_list=rewards,
                loss_masks=loss_masks,
                ratios=actor_segment_ratios,
                gamma=args.gamma,
            )

    elif args.advantage_estimator == "reinforce_plus_plus":
        rewards = torch.tensor(
            rewards, dtype=torch.float32, device=kl[0].device)
        returns = get_reinforce_plus_plus_returns(
            rewards=rewards,
            kl=kl,
            loss_masks=loss_masks,
            response_lengths=response_lengths,
            total_lengths=total_lengths,
            kl_coef=args.kl_coef,
            gamma=args.gamma,
        )
        advantages = [r for r in returns]

    elif args.advantage_estimator == "reinforce_plus_plus_baseline":
        rewards = torch.tensor(
            rewards, dtype=torch.float32, device=kl[0].device)
        advantages = get_reinforce_plus_plus_baseline_advantages(
            rewards=rewards,
            kl=kl,
            loss_masks=loss_masks,
            kl_coef=args.kl_coef,
        )
        returns = advantages

    else:
        raise NotImplementedError(
            f"advantage_estimator {args.advantage_estimator} is not supported. ")

    hlg_advantage_mode = getattr(args, "hlg_advantage_mode", "none")
    if hlg_advantage_mode != "none":
        if not _is_hl_gauss_value_loss(args):
            raise ValueError(
                f"--hlg-advantage-mode={hlg_advantage_mode} requires --critic-value-loss-type=hl_gauss")
        if args.advantage_estimator != "ppo":
            raise ValueError(
                f"--hlg-advantage-mode={hlg_advantage_mode} currently requires --advantage-estimator=ppo")
        if hlg_advantage_mode == "risk_baseline":
            beta = float(getattr(args, "hlg_adv_beta", 0.0))
            if beta < 0:
                raise ValueError(
                    f"hlg_adv_beta must be non-negative, got {beta}")
            if beta != 0:
                std_list = _validate_hlg_advantage_std(
                    mode="risk_baseline",
                    advantages=advantages,
                    value_std=value_std,
                )
                adjusted_advantages = []
                for adv, std in zip(advantages, std_list, strict=False):
                    adjusted_advantages.append(adv - beta * std)
                advantages = adjusted_advantages
        elif hlg_advantage_mode == "uncertainty_weighted":
            advantages = _apply_hlg_uncertainty_weighted_advantages(
                args,
                advantages=advantages,
                value_std=value_std,
            )
        else:
            raise ValueError(
                f"Unsupported hlg_advantage_mode: {hlg_advantage_mode}")

    # Apply on-policy distillation KL penalty to advantages (orthogonal to advantage estimator)
    if args.use_opd:
        apply_opd_kl_to_advantages(
            args=args,
            rollout_data=rollout_data,
            advantages=advantages,
            student_log_probs=log_probs,
        )

    # TODO: OpenRLHF always does advantages normalization but veRL doesn't seem to do it.
    if args.normalize_advantages:
        all_advs = torch.cat(advantages)
        cp_size = mpu.get_context_parallel_world_size()
        if cp_size == 1:
            all_masks = torch.cat(loss_masks)
        else:
            mask_chunks = []
            for i in range(len(advantages)):
                total_len = total_lengths[i]
                response_len = response_lengths[i]
                prompt_len = total_len - response_len
                max_seq_len = max_seq_lens[i] if max_seq_lens is not None else None

                _, _, _, token_offsets = get_logits_and_tokens_offset_with_cp(
                    total_len, response_len, args.qkv_format, max_seq_len
                )

                # Convert global offsets to response-space offsets
                s0, e0 = token_offsets[0]
                s1, e1 = token_offsets[1]
                res_s0, res_e0 = max(
                    0, s0 - prompt_len), max(0, e0 - prompt_len)
                res_s1, res_e1 = max(
                    0, s1 - prompt_len), max(0, e1 - prompt_len)

                local_mask_parts = []
                full_mask = loss_masks[i]
                if res_e0 > res_s0:
                    local_mask_parts.append(full_mask[res_s0:res_e0])
                if res_e1 > res_s1:
                    local_mask_parts.append(full_mask[res_s1:res_e1])

                # Concatenate the parts to form the final mask chunk for this rank and this sequence
                local_mask_chunk = (
                    torch.cat(local_mask_parts)
                    if local_mask_parts
                    else torch.tensor([], device=all_advs.device, dtype=full_mask.dtype)
                )
                mask_chunks.append(local_mask_chunk)

            all_masks = torch.cat(mask_chunks)

        if all_masks.numel() > 0:
            assert (
                all_advs.size() == all_masks.size()
            ), f"Shape mismatch before whitening: advantages {all_advs.size()}, masks {all_masks.size()}"
            dp_group = mpu.get_data_parallel_group()

            whitened_advs_flat = distributed_masked_whiten(
                all_advs,
                all_masks,
                process_group=dp_group,
                shift_mean=True,
            )
            chunk_lengths = [chunk.size(0) for chunk in advantages]
            advantages = list(torch.split(whitened_advs_flat, chunk_lengths))

    rollout_data["advantages"] = advantages
    rollout_data["returns"] = returns


def vanilla_tis_function(
    args,
    *,
    pg_loss: torch.Tensor,
    train_log_probs: list[torch.Tensor],
    rollout_log_probs: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
    **kwargs: Any,
) -> tuple[torch.Tensor, list[torch.Tensor], dict[str, torch.Tensor]]:
    rollout_log_probs = torch.cat(rollout_log_probs, dim=0)
    old_log_probs = torch.cat(train_log_probs, dim=0)
    tis = torch.exp(old_log_probs - rollout_log_probs)
    tis_abs = (torch.exp(old_log_probs - rollout_log_probs) - 1).abs()
    tis_weights = torch.clamp(tis, min=args.tis_clip_low, max=args.tis_clip)
    tis_clipfrac = (tis_weights != tis).float()
    metrics = {
        "tis": tis.clone().detach(),
        "tis_clipfrac": tis_clipfrac.clone().detach(),
        "tis_abs": tis_abs.clone().detach(),
    }
    pg_loss = pg_loss * tis_weights
    return pg_loss, loss_masks, metrics


def icepop_function(
    args,
    *,
    pg_loss: torch.Tensor,
    train_log_probs: list[torch.Tensor],
    rollout_log_probs: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
    **kwargs: Any,
) -> tuple[torch.Tensor, list[torch.Tensor], dict[str, torch.Tensor]]:
    rollout_log_probs = torch.cat(rollout_log_probs, dim=0)
    old_log_probs = torch.cat(train_log_probs, dim=0)
    ice_ratio = torch.exp(old_log_probs - rollout_log_probs)
    ice_abs = (torch.exp(old_log_probs - rollout_log_probs) - 1).abs()
    ice_weight = torch.where(
        (ice_ratio >= args.tis_clip_low) & (ice_ratio <=
                                            args.tis_clip), ice_ratio, torch.zeros_like(ice_ratio)
    )
    ice_clipfrac = (ice_weight != ice_ratio).float()
    metrics = {
        "tis": ice_ratio.clone().detach(),
        "tis_clipfrac": ice_clipfrac.clone().detach(),
        "tis_abs": ice_abs.clone().detach(),
    }
    pg_loss = pg_loss * ice_weight
    return pg_loss, loss_masks, metrics


def policy_loss_function(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute policy loss (PPO/GSPO) and metrics.

    Computes current log-probabilities and entropy from model logits, then
    calculates PPO-style clipped policy gradient loss. For GSPO, gathers
    full sequences via context-parallel all-gather before computing per-sample
    KL. Optionally applies TIS (Truncated Importance Sampling) correction and
    adds KL loss term if configured.

    Args:
        args: Configuration controlling advantage estimator, clipping thresholds,
            entropy/KL coefficients, and TIS settings.
        batch: Mini-batch containing "advantages", "log_probs" (old policy),
            "unconcat_tokens", "response_lengths", "total_lengths", "loss_masks",
            and optionally "ref_log_probs" and "rollout_log_probs".
        logits: Policy logits with shape `[1, T, V]`.
        sum_of_sample_mean: Reduction function that averages per-sample values.

    Returns:
        Tuple of `(loss, metrics)` where `loss` is a scalar tensor and `metrics`
        is a dict containing detached scalars: "loss", "pg_loss",
        "entropy_loss", "pg_clipfrac", "ppo_kl". Additional keys "kl_loss",
        "tis", "ois", "tis_clipfrac" are included when the respective features
        are enabled.
    """
    advantages = torch.cat(batch["advantages"], dim=0)
    old_log_probs = batch["rollout_log_probs"] if args.use_rollout_logprobs else batch["log_probs"]

    response_lengths = batch["response_lengths"]
    total_lengths = batch["total_lengths"]
    max_seq_lens = batch.get("max_seq_lens", None)

    _, log_probs_and_entropy = get_log_probs_and_entropy(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        with_entropy=True,
        max_seq_lens=max_seq_lens,
    )

    log_probs = log_probs_and_entropy["log_probs"]

    # Pre-gather log probs if needed by OPSM or GSPO to avoid duplicate gathering
    need_full_log_probs = args.use_opsm or args.advantage_estimator == "gspo"

    full_log_probs = None
    full_old_log_probs = None
    if need_full_log_probs:
        full_log_probs = [
            all_gather_with_cp(log_prob, total_length, response_length)
            for log_prob, total_length, response_length in zip(
                log_probs, total_lengths, response_lengths, strict=False
            )
        ]
        full_old_log_probs = [
            all_gather_with_cp(old_log_prob, total_length, response_length)
            for old_log_prob, total_length, response_length in zip(
                old_log_probs, total_lengths, response_lengths, strict=False
            )
        ]

    # Compute OPSM mask if enabled
    if args.use_opsm:
        opsm_mask, opsm_clipfrac = compute_opsm_mask(
            args=args,
            full_log_probs=full_log_probs,
            full_old_log_probs=full_old_log_probs,
            advantages=batch["advantages"],
            loss_masks=batch["loss_masks"],
        )

    # Compute KL divergence (GSPO uses sequence-level KL, others use per-token KL)
    if args.advantage_estimator == "gspo":
        ppo_kl = compute_gspo_kl(
            full_log_probs=full_log_probs,
            full_old_log_probs=full_old_log_probs,
            local_log_probs=log_probs,
            loss_masks=batch["loss_masks"],
        )
        old_log_probs = torch.cat(old_log_probs, dim=0)
        log_probs = torch.cat(log_probs, dim=0)
    else:
        old_log_probs = torch.cat(old_log_probs, dim=0)
        log_probs = torch.cat(log_probs, dim=0)
        ppo_kl = old_log_probs - log_probs

    pg_loss, pg_clipfrac = compute_policy_loss(
        ppo_kl,
        advantages,
        args.eps_clip,
        args.eps_clip_high,
        getattr(args, "eps_clip_c", None),
    )

    if args.use_opsm:
        pg_loss = pg_loss * opsm_mask

    # Apply off-policy correction using importance sampling if enabled
    if args.get_mismatch_metrics or args.use_tis:
        # NOTE:
        # `tis_func` may apply rejection-sampling style masking (RS) and return `modified_response_masks`.
        # We rebuild `sum_of_sample_mean` with those masks to correct denominators for loss/backprop.
        #
        # However, mismatch/TIS/RS metrics (e.g., "truncate_fraction") are often defined over the
        # *pre-RS* valid tokens. If we aggregate metrics with `modified_response_masks`, the rejected
        # tokens are excluded from the denominator and the metric can be artificially driven to 0.
        # Keep a copy of the original reducer (based on `batch["loss_masks"]`) for metric aggregation.
        sum_of_sample_mean_for_mismatch_metrics = sum_of_sample_mean

        assert "rollout_log_probs" in batch, "rollout_log_probs must be provided for TIS"

        ois = (-ppo_kl).exp()
        tis_kwargs = {
            "args": args,
            "pg_loss": pg_loss,
            "train_log_probs": batch["log_probs"],
            "rollout_log_probs": batch["rollout_log_probs"],
            "loss_masks": batch["loss_masks"],
            "total_lengths": total_lengths,
            "response_lengths": response_lengths,
        }

        if args.custom_tis_function_path is not None:
            tis_func = load_function(args.custom_tis_function_path)
        else:
            tis_func = vanilla_tis_function
        pg_loss, modified_response_masks, tis_metrics = tis_func(**tis_kwargs)

        # [decouple IS and rejection] Rebuild sum_of_sample_mean with modified_response_masks for denominator correction
        # modified_response_masks will be sliced with cp in get_sum_of_sample_mean
        sum_of_sample_mean = get_sum_of_sample_mean(
            total_lengths,
            response_lengths,
            modified_response_masks,
            sample_weights=batch.get("sample_weights"),
            calculate_per_token_loss=args.calculate_per_token_loss,
            qkv_format=args.qkv_format,
            max_seq_lens=max_seq_lens,
        )

    # Determine pg_loss reducer: use custom if specified, otherwise default
    if getattr(args, "ppo_token_level_loss", False):
        pg_loss_masks = modified_response_masks if (
            args.get_mismatch_metrics or args.use_tis) else batch["loss_masks"]
        token_sum = get_sum_of_sample_mean(
            total_lengths, response_lengths, pg_loss_masks,
            calculate_per_token_loss=True, qkv_format=args.qkv_format, max_seq_lens=max_seq_lens,
        )

        def pg_loss_reducer(x):
            return normalize_vapo_token_sum(token_sum(x), batch["vapo_loss_normalizers"])

    elif getattr(args, "custom_pg_loss_reducer_function_path", None) is not None:
        custom_pg_loss_reducer_func = load_function(
            args.custom_pg_loss_reducer_function_path)
        # Determine which loss_masks to use for pg_loss reducer
        pg_loss_masks = modified_response_masks if (
            args.get_mismatch_metrics or args.use_tis) else batch["loss_masks"]
        pg_loss_reducer = custom_pg_loss_reducer_func(
            total_lengths, response_lengths, pg_loss_masks, args.calculate_per_token_loss
        )
    else:
        pg_loss_reducer = sum_of_sample_mean

    pg_loss = pg_loss_reducer(pg_loss)
    pg_clipfrac = sum_of_sample_mean(pg_clipfrac)
    ppo_kl = sum_of_sample_mean(ppo_kl)

    # entropy loss
    entropy = log_probs_and_entropy["entropy"]
    entropy = torch.cat(entropy, dim=0)
    entropy_loss = sum_of_sample_mean(entropy)

    loss = pg_loss - args.entropy_coef * entropy_loss

    positive_nll_loss = None
    if getattr(args, "use_positive_nll_loss", False):
        positive_masks = positive_example_masks(
            batch.get("positive_nll_rewards"), batch["loss_masks"], args.positive_reward_threshold,
        )
        positive_token_sum = get_sum_of_sample_mean(
            total_lengths, response_lengths, positive_masks,
            calculate_per_token_loss=True, qkv_format=args.qkv_format, max_seq_lens=max_seq_lens,
        )
        # This is supervised NLL on verifier-positive responses, without PPO/TIS
        # weighting. Normalize across the entire optimizer step, not each micro-batch.
        positive_nll_loss = normalize_vapo_token_sum(
            positive_token_sum(-log_probs), batch["vapo_loss_normalizers"], positive=True,
        )
        loss = loss + args.positive_nll_coef * positive_nll_loss

    if args.use_kl_loss:
        ref_log_probs = batch["ref_log_probs"]
        ref_log_probs = torch.cat(ref_log_probs, dim=0)
        importance_ratio = None
        if args.use_unbiased_kl:
            importance_ratio = torch.exp(log_probs - old_log_probs)
        kl = compute_approx_kl(
            log_probs,
            ref_log_probs,
            kl_loss_type=args.kl_loss_type,
            importance_ratio=importance_ratio,
        )
        kl_loss = sum_of_sample_mean(kl)

        loss = loss + args.kl_loss_coef * kl_loss

    # make sure the gradient could backprop correctly.
    if log_probs.numel() == 0:
        loss += 0 * logits.sum()

    train_rollout_logprob_abs_diff = None
    if "rollout_log_probs" in batch and batch["rollout_log_probs"]:
        rollout_log_probs = torch.cat(batch["rollout_log_probs"], dim=0)
        train_rollout_logprob_abs_diff = sum_of_sample_mean(
            (old_log_probs - rollout_log_probs).abs())

    reported_loss = {
        "loss": loss.clone().detach(),
        "pg_loss": pg_loss.clone().detach(),
        "entropy_loss": entropy_loss.clone().detach(),
        "pg_clipfrac": pg_clipfrac.clone().detach(),
        "ppo_kl": ppo_kl.clone().detach(),
    }

    if positive_nll_loss is not None:
        reported_loss["positive_nll_loss"] = positive_nll_loss.detach().clone()
        reported_loss["positive_nll_loss_scaled"] = args.positive_nll_coef * positive_nll_loss.detach().clone()

    if train_rollout_logprob_abs_diff is not None:
        reported_loss["train_rollout_logprob_abs_diff"] = train_rollout_logprob_abs_diff.clone(
        ).detach()

    if args.use_kl_loss:
        reported_loss["kl_loss"] = kl_loss.clone().detach()

    if args.get_mismatch_metrics or args.use_tis:
        # Aggregate mismatch/TIS/RS related metrics with the *pre-RS* masks.
        # See comment above where `sum_of_sample_mean_for_mismatch_metrics` is defined.
        reported_loss["ois"] = sum_of_sample_mean_for_mismatch_metrics(
            ois).clone().detach()
        # Assume all metrics are already cloned and detached
        for metric_key, metric_value in tis_metrics.items():
            key_name = f"{metric_key}"
            reported_loss[key_name] = sum_of_sample_mean_for_mismatch_metrics(
                metric_value)

    if args.use_opsm:
        reported_loss["opsm_clipfrac"] = opsm_clipfrac

    # Add OPD metrics if available
    if "opd_reverse_kl" in batch:
        opd_reverse_kl = torch.cat(batch["opd_reverse_kl"], dim=0)
        reported_loss["opd_reverse_kl"] = sum_of_sample_mean(
            opd_reverse_kl).clone().detach()

    return loss, reported_loss


def value_loss_function(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute value loss and metrics.

    Extracts current value predictions from `logits`, compares them against
    returns, and applies PPO-style value clipping to scalar value predictions.
    TPPO values in logit space instead use BCE with optional logit clipping.

    Args:
        args: Configuration containing `value_clip` threshold.
        batch: Mini-batch with "values" (old predictions), "returns",
            "unconcat_tokens", "total_lengths", and "response_lengths".
        logits: Value head output with shape `[1, T, 1]`.
        sum_of_sample_mean: Reduction function that averages per-sample values.

    Returns:
        Tuple of `(loss, metrics)` where `loss` is a scalar tensor and
        `metrics` contains detached scalars "value_loss" and "value_clipfrac".
    """
    use_dense_critic_only_warmup = _in_dense_critic_only_warmup(args, batch)
    use_mc_value_labels = _use_mc_value_labels(
        args) and not use_dense_critic_only_warmup
    returns_chunks = _get_value_target_chunks(args, batch)
    returns = torch.cat(returns_chunks, dim=0)
    value_loss_masks = _get_value_loss_masks(args, batch)
    use_last_value_only = (
        _use_separate_v0_head(args)
        and bool(getattr(args, "critic_separate_v0_head_last_token_only", False))
    )
    if use_last_value_only:
        value_loss_masks = _select_last_valid_value_loss_masks(value_loss_masks)
        value_loss_reducer = get_sum_of_sample_mean(
            batch["total_lengths"],
            batch["response_lengths"],
            value_loss_masks,
            sample_weights=batch.get("sample_weights"),
            calculate_per_token_loss=args.calculate_per_token_loss,
            qkv_format=args.qkv_format,
            max_seq_lens=batch.get("max_seq_lens"),
        )
    else:
        value_loss_reducer = sum_of_sample_mean
    hlg_target_clipped_frac = None
    hlg_target_mean_mse = None
    hlg_return_mse = None
    hlg_target_sigma_mean = None
    raw_value_logits = []
    value_logits_list = []
    v0_logit_list = []
    v0_value_list = []
    tppo_deltas = []

    if _is_hl_gauss_value_loss(args):
        value_logits = []
        hlg_num_bins = int(getattr(args, "hlg_num_bins", 101))
        old_values = torch.cat(batch["values"], dim=0)
        for logits_chunk, _ in get_responses(
            logits,
            args=args,
            unconcat_tokens=batch["unconcat_tokens"],
            total_lengths=batch["total_lengths"],
            response_lengths=batch["response_lengths"],
            max_seq_lens=batch.get("max_seq_lens", None),
        ):
            if logits_chunk.size(-1) != hlg_num_bins:
                raise ValueError(
                    f"HL-Gauss expects value head dim={hlg_num_bins}, but got logits chunk shape {tuple(logits_chunk.shape)}"
                )
            value_logits.append(logits_chunk)

        if args.allgather_cp:
            logits_dict = {"value_logits": value_logits}
            _allgather_cp_redistribute(
                logits_dict,
                logits=logits,
                args=args,
                total_lengths=batch["total_lengths"],
                response_lengths=batch["response_lengths"],
                max_seq_lens=batch.get("max_seq_lens", None),
            )
            value_logits = logits_dict["value_logits"]

        value_logits = torch.cat([x.reshape(-1, x.size(-1))
                                 for x in value_logits], dim=0)
        if value_logits.size(0) != returns.size(0):
            raise ValueError(
                f"HL-Gauss logits/returns length mismatch: logits={value_logits.size(0)} vs returns={returns.size(0)}"
            )

        hlg_target_mode = getattr(args, "hlg_target_mode", "single")
        group_success_p = None
        if hlg_target_mode == "group_bernoulli_mixture":
            group_target_weight = float(
                getattr(args, "hlg_group_target_weight", 0.0))
            if group_target_weight < 0 or group_target_weight > 1:
                raise ValueError(
                    f"hlg_group_target_weight must be in [0, 1], got {group_target_weight}")
            sample_group_success_p = batch.get("hlg_group_success_p")
            if sample_group_success_p is None:
                raise ValueError(
                    "--hlg-target-mode=group_bernoulli_mixture requires rollout_data['hlg_group_success_p']"
                )

            group_success_p = _expand_sample_scalars_to_token_values(
                sample_group_success_p,
                returns_chunks,
                device=returns.device,
                dtype=torch.float32,
            ).clamp(0.0, 1.0)
        elif hlg_target_mode == "loo_position_beta":
            pass
        elif hlg_target_mode != "single":
            raise ValueError(f"Unsupported hlg_target_mode: {hlg_target_mode}")

        if hlg_target_mode == "loo_position_beta":
            target_sigma = None
            target_clipped_mask = torch.zeros_like(returns, dtype=torch.bool)
            target_probs = _hl_loo_position_beta_target_probs(
                args,
                returns_chunks=returns_chunks,
                returns=returns,
                response_lengths=batch["response_lengths"],
                rewards=batch.get("rewards"),
                group_num_correct=batch.get("hlg_group_num_correct"),
                group_size=batch.get("hlg_group_size"),
                dtype=value_logits.dtype,
            )
        else:
            target_sigma = _hl_gauss_target_sigma_from_context(
                args,
                returns_chunks=returns_chunks,
                returns=returns,
                response_lengths=batch["response_lengths"],
                group_success_p=group_success_p,
                mc_value_stds=batch.get("mc_value_stds"),
                mc_value_ns=batch.get("mc_value_ns"),
                use_mc_value_labels=use_mc_value_labels,
            )
            target_probs, target_clipped_mask = _hl_gauss_target_probs_from_values(
                returns,
                args,
                dtype=value_logits.dtype,
                sigma=target_sigma,
            )

            if hlg_target_mode == "group_bernoulli_mixture":
                v_min = float(getattr(args, "hlg_v_min", 0.0))
                v_max = float(getattr(args, "hlg_v_max", 1.0))
                low_values = torch.full_like(returns, v_min)
                high_values = torch.full_like(returns, v_max)
                low_target_probs, _ = _hl_gauss_target_probs_from_values(
                    low_values,
                    args,
                    dtype=value_logits.dtype,
                    sigma=target_sigma,
                )
                high_target_probs, _ = _hl_gauss_target_probs_from_values(
                    high_values,
                    args,
                    dtype=value_logits.dtype,
                    sigma=target_sigma,
                )
                group_success_p = group_success_p.to(
                    device=value_logits.device, dtype=value_logits.dtype).unsqueeze(-1)
                group_target_probs = (
                    1.0 - group_success_p) * low_target_probs + group_success_p * high_target_probs
                target_probs = (1.0 - group_target_weight) * \
                    target_probs + group_target_weight * group_target_probs
                target_probs = target_probs / \
                    target_probs.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            if target_sigma is not None:
                hlg_target_sigma_mean = target_sigma.to(
                    device=value_logits.device, dtype=value_logits.dtype)
        ce_loss = -(target_probs *
                    F.log_softmax(value_logits, dim=-1)).sum(dim=-1)
        values = _hl_gauss_expected_from_logits(value_logits, args)
        target_mean = _hl_gauss_expected_from_probs(target_probs, args).to(
            device=values.device, dtype=values.dtype)

        clip_aux_target_mode = getattr(args, "hlg_clip_aux_target", "returns")
        if clip_aux_target_mode == "returns":
            clip_aux_target = returns.to(
                device=values.device, dtype=values.dtype)
        elif clip_aux_target_mode == "target_mean":
            clip_aux_target = target_mean
        else:
            raise ValueError(
                f"Unsupported hlg_clip_aux_target: {clip_aux_target_mode}")
        hlg_target_mean_mse = (values - target_mean) ** 2
        hlg_return_mse = (
            values - returns.to(device=values.device, dtype=values.dtype)) ** 2

        # Simplified HL-Gauss clipping: keep CE as main objective, add PPO-style
        # clipping auxiliary loss in scalar expectation space.
        if args.value_clip > 0:
            values_clipfrac = torch.abs(values - old_values) > args.value_clip
            values_clipped = old_values + \
                (values - old_values).clamp(-args.value_clip, args.value_clip)
            clip_surr1 = (values_clipped - clip_aux_target) ** 2
            clip_surr2 = (values - clip_aux_target) ** 2
            clip_aux_loss = torch.max(clip_surr1, clip_surr2)
        else:
            values_clipfrac = torch.zeros_like(values, dtype=torch.bool)
            clip_aux_loss = (values - clip_aux_target) ** 2

        clip_aux_coef = float(getattr(args, "hlg_clip_aux_coef", 1.0))
        loss = ce_loss + clip_aux_coef * clip_aux_loss
        hlg_target_clipped_frac = target_clipped_mask.float()
    else:
        old_values = torch.cat(batch["values"], dim=0)
        _, values = get_values(
            logits,
            args=args,
            unconcat_tokens=batch["unconcat_tokens"],
            total_lengths=batch["total_lengths"],
            response_lengths=batch["response_lengths"],
            max_seq_lens=batch.get("max_seq_lens", None),
        )
        raw_value_logits = values.get("critic_raw_value_logits", [])
        value_logits_list = values.get("critic_value_logits", [])
        v0_logit_list = values.get("critic_v0_logits", [])
        v0_value_list = values.get("critic_v0_values", [])
        tppo_deltas = values.get("critic_tppo_deltas", [])
        values = torch.cat([value.flatten()
                           for value in values["values"]], dim=0)

        use_separate_bce = _use_separate_v0_head(args)
        use_delta_bce = (
            _learn_tppo(args)
            and not _use_tppo_head_space(args)
            and not use_separate_bce
        )
        use_sigmoid_bce = _use_sigmoid_bce_value(args)
        if use_separate_bce or use_delta_bce or use_sigmoid_bce:
            if not value_logits_list:
                raise ValueError(
                    "logit-space BCE value loss requires critic value logits.")
            value_logits = torch.cat([value.flatten()
                                     for value in value_logits_list], dim=0)
            targets = returns.clamp(min=0.0, max=1.0).to(
                dtype=value_logits.dtype)
            loss = F.binary_cross_entropy_with_logits(
                value_logits, targets, reduction="none")
            # Separate-v0 BCE uses reconstructed logits directly. The ordinary
            # scalar/delta variants optionally clip in their logit spaces.
            value_logit_clip = 0.0 if use_separate_bce else (
                _sigmoid_bce_value_logit_clip(args)
                if use_sigmoid_bce
                else _tppo_logit_clip(args)
            )
            value_logit_clipfrac = torch.zeros_like(values, dtype=torch.bool)
            if value_logit_clip > 0:
                old_value_logits = torch.logit(
                    old_values.clamp(min=1e-6, max=1.0 - 1e-6)
                ).to(dtype=value_logits.dtype)
                logit_delta = value_logits - old_value_logits
                value_logit_clipfrac = logit_delta.abs() > value_logit_clip
                value_logits_clipped = old_value_logits + \
                    logit_delta.clamp(-value_logit_clip, value_logit_clip)
                loss_clipped = F.binary_cross_entropy_with_logits(
                    value_logits_clipped, targets, reduction="none")
                loss = torch.max(loss, loss_clipped)
            values_clipfrac = value_logit_clipfrac
        else:
            loss, values_clipfrac = _clipped_value_mse(
                values, old_values, returns, args.value_clip)

    # This metric deliberately uses the original 0/1 return, even when the
    # value loss itself is supervised by MC labels.
    value_returns = torch.cat(batch["returns"], dim=0).to(
        device=values.device, dtype=values.dtype)
    value_metric_mask = _get_local_value_loss_mask(args, batch, value_loss_masks).to(
        device=values.device, dtype=values.dtype)
    value_loss_mask_count = value_metric_mask.sum()
    value_explained_variance_stats = compute_masked_explained_variance_stats(
        value_returns,
        values,
        value_metric_mask,
    )

    loss = value_loss_reducer(loss)
    value_loss_base = loss
    values_clipfrac = value_loss_reducer(values_clipfrac.float())

    if hlg_target_clipped_frac is not None:
        hlg_target_clipped_frac = sum_of_sample_mean(
            hlg_target_clipped_frac.float())
    if hlg_target_mean_mse is not None:
        hlg_target_mean_mse = sum_of_sample_mean(hlg_target_mean_mse)
    if hlg_return_mse is not None:
        hlg_return_mse = sum_of_sample_mean(hlg_return_mse)
    if hlg_target_sigma_mean is not None:
        hlg_target_sigma_mean = sum_of_sample_mean(hlg_target_sigma_mean)

    # make sure the gradient could backprop correctly.
    if values.numel() == 0:
        loss += 0 * values.sum()

    reported_loss = {
        "value_loss": loss.clone().detach(),
        "value_loss_base": value_loss_base.clone().detach(),
        "value_clipfrac": values_clipfrac.clone().detach(),
        "value_loss_mask_count": value_loss_mask_count.clone().detach(),
    }
    reported_loss.update(
        {
            f"{VALUE_EXPLAINED_VARIANCE_STAT_PREFIX}{name}": value.clone().detach()
            for name, value in value_explained_variance_stats.items()
        }
    )
    if (
        _learn_tppo(args)
        and not _use_tppo_head_space(args)
        and _tppo_logit_clip(args) > 0
    ) or (
        _use_sigmoid_bce_value(args) and _sigmoid_bce_value_logit_clip(args) > 0
    ):
        reported_loss["value_logit_clipfrac"] = values_clipfrac.clone(
        ).detach()
    if raw_value_logits:
        reported_loss.update(_stats_for_tensor_list(
            "critic_raw_value_logits", raw_value_logits))
    if value_logits_list:
        reported_loss.update(_stats_for_tensor_list(
            "critic_value_logits", value_logits_list))
    if v0_logit_list:
        reported_loss.update(_stats_for_tensor_list(
            "critic_v0_logits", v0_logit_list))
    if v0_value_list:
        reported_loss.update(_stats_for_tensor_list(
            "critic_v0_values", v0_value_list))
    if not _is_hl_gauss_value_loss(args) and values.numel() == sum(int(x) for x in batch["response_lengths"]):
        reported_loss.update(
            _stats_for_tensor_list(
                "critic_values",
                [value.flatten() for value in values.split(
                    tuple(batch["response_lengths"]), dim=0)],
            )
        )
    if tppo_deltas:
        reported_loss.update(_stats_for_tensor_list(
            "critic_tppo_deltas", tppo_deltas))
    if hlg_target_clipped_frac is not None:
        reported_loss["hlg_target_clipped_frac"] = hlg_target_clipped_frac.clone(
        ).detach()
    if hlg_target_mean_mse is not None:
        reported_loss["hlg_target_mean_mse"] = hlg_target_mean_mse.clone().detach()
    if hlg_return_mse is not None:
        reported_loss["hlg_return_mse"] = hlg_return_mse.clone().detach()
    if hlg_target_sigma_mean is not None:
        reported_loss["hlg_target_sigma_mean"] = hlg_target_sigma_mean.clone(
        ).detach()

    return loss, reported_loss


def sft_loss_function(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute supervised fine-tuning loss over response tokens.

    Computes log-probabilities of the ground-truth tokens in the response
    segments and returns the negative log-likelihood as the loss.

    Args:
        args: Configuration (passed through to helpers).
        batch: Mini-batch with "unconcat_tokens", "response_lengths", and
            "total_lengths".
        logits: Policy logits with shape `[1, T, V]`.
        sum_of_sample_mean: Reduction function that averages per-sample values.

    Returns:
        Tuple of `(loss, metrics)` where `metrics` contains a single detached
        scalar "loss".
    """
    response_lengths = batch["response_lengths"]
    total_lengths = batch["total_lengths"]

    _, log_probs_and_entropy = get_log_probs_and_entropy(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        with_entropy=False,
        max_seq_lens=batch.get("max_seq_lens", None),
    )

    log_probs = log_probs_and_entropy["log_probs"]
    log_probs = torch.cat(log_probs, dim=0)
    loss = -sum_of_sample_mean(log_probs)

    # make sure the gradient could backprop correctly.
    if log_probs.numel() == 0:
        loss += 0 * logits.sum()

    return (
        loss,
        {
            "loss": loss.clone().detach(),
        },
    )


def _scale_and_pack_megatron_loss(
    args: Namespace,
    loss: torch.Tensor,
    log: dict[str, torch.Tensor],
    logits: torch.Tensor,
    num_tokens: torch.Tensor,
    report_count: int | float | torch.Tensor,
    num_microbatches: int,
    global_batch_size: int,
) -> tuple[torch.Tensor, int | torch.Tensor, dict[str, list[str] | torch.Tensor]]:
    # With allgather-CP, some CP ranks may have no loss-contributing tokens (e.g., all
    # padding). Without this, gradient doesn't flow through their attention path, so
    # the CP gather's backward (reduce-scatter) is not called, deadlocking other CP
    # ranks that call it. Adding this zero loss forces autograd to traverse the full
    # graph on every rank without changing gradient values.
    if args.allgather_cp and mpu.get_context_parallel_world_size() > 1:
        loss = loss + 0 * logits.sum()

    # Here we need to divide by cp_size because to cancel the multiply in Megatron.
    if not args.calculate_per_token_loss:
        loss = (
            loss * num_microbatches / global_batch_size *
            mpu.get_data_parallel_world_size(with_context_parallel=True)
        )
    else:
        loss = loss * mpu.get_context_parallel_world_size()

    return (
        loss,
        (num_tokens if args.calculate_per_token_loss else torch.tensor(
            1, device=logits.device)),
        {
            "keys": list(log.keys()),
            "values": torch.tensor(
                [
                    num_tokens if args.calculate_per_token_loss else report_count,
                ]
                + list(log.values()),
                device=logits.device,
            ),
        },
    )


def loss_function(
    args: Namespace,
    batch: RolloutBatch,
    num_microbatches: int,
    logits: torch.Tensor,
) -> tuple[torch.Tensor, int | torch.Tensor, dict[str, list[str] | torch.Tensor]]:
    """Dispatch to the configured loss and rescale for Megatron integration.

    Selects one of "policy_loss", "value_loss", "sft_loss", or a custom loss
    function based on `args.loss_type`, computes the loss and metrics, then
    rescales the loss by micro-batch and parallelism factors to integrate with
    Megatron's gradient accumulation.

    Args:
        args: Configuration specifying `loss_type`, `calculate_per_token_loss`,
            `global_batch_size`, and optionally `custom_loss_function_path`.
        batch: Mini-batch with "loss_masks", "response_lengths", and other
            keys required by the selected loss function.
        num_microbatches: Number of gradient accumulation steps.
        logits: Model outputs (policy or value head).

    Returns:
        Tuple of `(scaled_loss, normalizer, logging_dict)` where:
        - `scaled_loss` is the loss tensor (scalar) rescaled for Megatron.
        - `normalizer` is `num_tokens` (scalar tensor) if
          `args.calculate_per_token_loss` is True, else `1` (int).
        - `logging_dict` has keys "keys" (list of str metric names) and
          "values" (1D tensor: [count, metric1, metric2, ...]).
    """
    loss_type = args.loss_type
    loss_masks_for_reducer = batch["loss_masks"]
    if loss_type == "value_loss":
        loss_masks_for_reducer = _get_value_loss_masks(args, batch)

    num_tokens = sum([torch.clamp_min(loss_mask.sum(), 1)
                     for loss_mask in loss_masks_for_reducer])
    num_samples = len(batch["response_lengths"])
    sample_weights = _get_sample_weights_for_loss(args, batch)

    sum_of_sample_mean = get_sum_of_sample_mean(
        batch["total_lengths"],
        batch["response_lengths"],
        loss_masks_for_reducer,
        sample_weights=sample_weights,
        calculate_per_token_loss=args.calculate_per_token_loss,
        qkv_format=args.qkv_format,
        max_seq_lens=batch.get("max_seq_lens", None),
    )

    match loss_type:
        case "policy_loss":
            func = policy_loss_function
        case "value_loss":
            func = value_loss_function
        case "sppo_value_loss":
            from .sppo import sppo_value_loss_function

            func = sppo_value_loss_function
        case "sft_loss":
            func = sft_loss_function
        case "custom_loss":
            func = load_function(args.custom_loss_function_path)
        case _:
            raise ValueError(f"Unknown loss type: {loss_type}")

    if args.recompute_loss_function:
        loss, log = checkpoint(func, args, batch, logits, sum_of_sample_mean)
    else:
        loss, log = func(args, batch, logits, sum_of_sample_mean)

    return _scale_and_pack_megatron_loss(
        args=args,
        loss=loss,
        log=log,
        logits=logits,
        num_tokens=num_tokens,
        report_count=num_samples,
        num_microbatches=num_microbatches,
        global_batch_size=batch.get("dynamic_global_batch_size",
                                    args.global_batch_size),
    )
