# Adapt from https://github.com/OpenRLHF/OpenRLHF/blob/10c733694ed9fbb78a0a2ff6a05efc7401584d46/openrlhf/models/utils.py
# and https://github.com/OpenRLHF/OpenRLHF/blob/10c733694ed9fbb78a0a2ff6a05efc7401584d46/openrlhf/trainer/ppo_utils/experience_maker.py

from argparse import Namespace

import torch
import torch.distributed as dist
import torch.nn.functional as F


@torch.compile(dynamic=True)
def compute_approx_kl(
    log_probs: torch.Tensor,
    log_probs_base: torch.Tensor,
    kl_loss_type: str,
    importance_ratio: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Compute the approximate KL divergence between two distributions.
    Schulman blog: http://joschu.net/blog/kl-approx.html

    Args:
        log_probs: Log probabilities of the new distribution.
        log_probs_base: Log probabilities of the base distribution.
        kl_loss_type: Type of KL estimator (k1, k2, k3, low_var_kl).
        importance_ratio: Optional IS ratio (π_θ/π_old) for unbiased KL estimation.
    """
    log_ratio = log_probs.float() - log_probs_base.float()

    if kl_loss_type == "k1":
        kl = log_ratio
    elif kl_loss_type == "k2":
        kl = log_ratio**2 / 2.0
    elif kl_loss_type in ["k3", "low_var_kl"]:
        # The non negative kl approximation in
        # http://joschu.net/blog/kl-approx.html
        # Besides non negative, it is also unbiased and have lower variance.
        log_ratio = -log_ratio
        kl = log_ratio.exp() - 1 - log_ratio
    else:
        raise ValueError(f"Unknown kl_loss_type: {kl_loss_type}")

    # Apply IS ratio for unbiased KL estimation (DeepSeek-V3.2)
    if importance_ratio is not None:
        kl = importance_ratio * kl

    # Clamp only for low_var_kl for numerical stability
    if kl_loss_type == "low_var_kl":
        kl = torch.clamp(kl, min=-10, max=10)

    return kl


def compute_opsm_mask(
    args: Namespace,
    full_log_probs: list[torch.Tensor],
    full_old_log_probs: list[torch.Tensor],
    advantages: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute Off-Policy Sequence Masking (OPSM) mask.

    Args:
        args: Configuration containing `opsm_delta` threshold.
        full_log_probs: Current policy log-probs per sample.
        full_old_log_probs: Old policy log-probs per sample.
        advantages: Advantage values per sample.
        loss_masks: Loss masks per sample.

    Returns:
        Tuple of `(opsm_mask, opsm_clipfrac)` where `opsm_mask` is a
        concatenated tensor of per-token masks and
        `opsm_clipfrac` is the count of masked sequences.
    """
    opsm_mask_list = []
    device = advantages[0].device
    opsm_clipfrac = torch.tensor(0.0, device=device)

    for full_log_prob, full_old_log_prob, advantage, loss_mask in zip(
        full_log_probs, full_old_log_probs, advantages, loss_masks, strict=False
    ):
        # Calculate sequence-level KL
        seq_kl = ((full_old_log_prob - full_log_prob) * loss_mask).sum() / torch.clamp_min(loss_mask.sum(), 1)

        # Create mask: 0 if (advantage < 0 and seq_kl > delta), else 1
        mask = ((advantage < 0) & (seq_kl > args.opsm_delta)).float()
        opsm_clipfrac += mask.sum() / torch.clamp_min(loss_mask.sum(), 1)

        opsm_mask_list.append(1 - mask)

    opsm_mask = torch.cat(opsm_mask_list, dim=0)
    return opsm_mask, opsm_clipfrac


def compute_gspo_kl(
    full_log_probs: list[torch.Tensor],
    full_old_log_probs: list[torch.Tensor],
    local_log_probs: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
) -> torch.Tensor:
    """Compute GSPO-style per-sequence KL divergence.

    Args:
        full_log_probs: Current policy log-probs per sample (full or CP-local).
        full_old_log_probs: Old policy log-probs per sample (full or CP-local).
        local_log_probs: Local (CP-local) log-probs for expansion shape reference.
        loss_masks: Loss masks per sample.

    Returns:
        Concatenated tensor of per-token KL values where each token in a
        sequence has the same KL value (the sequence-level KL).
    """
    # Compute sequence-level KL and expand to per-token
    ppo_kl = [
        ((old_logprob - log_prob) * loss_mask).sum() / torch.clamp_min(loss_mask.sum(), 1)
        for log_prob, old_logprob, loss_mask in zip(full_log_probs, full_old_log_probs, loss_masks, strict=False)
    ]
    ppo_kl = [kl.expand_as(log_prob) for kl, log_prob in zip(ppo_kl, local_log_probs, strict=False)]
    ppo_kl = torch.cat(ppo_kl, dim=0)

    return ppo_kl


@torch.compile(dynamic=True)
def compute_policy_loss(
    ppo_kl: torch.Tensor,
    advantages: torch.Tensor,
    eps_clip: float,
    eps_clip_high: float,
    eps_clip_c: float | None = None,
):
    ratio = (-ppo_kl).exp()
    pg_losses1 = -ratio * advantages
    pg_losses2 = -ratio.clamp(1 - eps_clip, 1 + eps_clip_high) * advantages
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)
    clipfrac = torch.gt(pg_losses2, pg_losses1).float()

    if eps_clip_c is not None:
        assert (
            eps_clip_c > 1.0
        ), f"The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0, but get the value: {eps_clip_c}."
        pg_losses3 = -eps_clip_c * advantages
        clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
        pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
    else:
        pg_losses = clip_pg_losses1

    return pg_losses, clipfrac


def compute_log_probs(logits: torch.Tensor, tokens: torch.Tensor, process_group: dist.ProcessGroup | None):
    # TODO: when megatron is not installed, fall back to naive implementation
    from megatron.core.fusions.fused_cross_entropy import fused_vocab_parallel_cross_entropy

    # convert to [seq_len, batch_size, vocab_size] as expected by fused_vocab_parallel_cross_entropy
    logits = logits.unsqueeze(1)
    tokens = tokens.unsqueeze(1)
    return -fused_vocab_parallel_cross_entropy(logits, tokens, process_group)


# from https://github.com/volcengine/verl/blob/0bdf7f469854815177e73dcfe9e420836c952e6e/verl/utils/megatron/tensor_parallel.py#L99
class _VocabParallelEntropy(torch.autograd.Function):

    @staticmethod
    def forward(ctx, vocab_parallel_logits: torch.Tensor, process_group: dist.ProcessGroup) -> torch.Tensor:

        @torch.compile(dynamic=True)
        def mul_reduce(a, b):
            return (a * b).sum(dim=-1, keepdim=True)

        logits_max = vocab_parallel_logits.max(dim=-1, keepdim=True).values
        dist.all_reduce(logits_max, op=dist.ReduceOp.MAX, group=process_group)
        normalized_vocab_parallel_logits = vocab_parallel_logits - logits_max
        normalized_exp_logits = normalized_vocab_parallel_logits.exp_()
        normalized_sum_exp_logits = normalized_exp_logits.sum(dim=-1, keepdim=True)
        dist.all_reduce(normalized_sum_exp_logits, group=process_group)
        softmax_logits = normalized_exp_logits.div_(normalized_sum_exp_logits)
        sum_softmax_times_logits = mul_reduce(softmax_logits, vocab_parallel_logits)
        dist.all_reduce(sum_softmax_times_logits, group=process_group)
        entropy = logits_max + normalized_sum_exp_logits.log() - sum_softmax_times_logits
        ctx.save_for_backward(vocab_parallel_logits, softmax_logits, sum_softmax_times_logits)
        return entropy.squeeze(dim=-1)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        vocab_parallel_logits, softmax_logits, sum_softmax_times_logits = ctx.saved_tensors
        # reuse softmax_logits as grad
        vocab_parallel_logits.sub_(sum_softmax_times_logits)
        softmax_logits.mul_(vocab_parallel_logits)
        softmax_logits.mul_(grad_output.unsqueeze(dim=-1))
        # recover vocab_parallel_logits
        vocab_parallel_logits.add_(sum_softmax_times_logits)
        softmax_logits.mul_(-1)
        return softmax_logits, None


def compute_entropy_from_logits(logits: torch.Tensor, process_group) -> torch.Tensor:
    return _VocabParallelEntropy.apply(logits, process_group)


def get_grpo_returns(
    rewards: torch.Tensor,
    kl: list[torch.Tensor],
):
    returns = []
    for i in range(len(rewards)):
        returns.append(torch.ones_like(kl[i]) * rewards[i])
    return returns


def get_reinforce_plus_plus_returns(
    rewards: torch.Tensor,
    kl: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
    response_lengths: list[int],
    total_lengths: list[int],
    kl_coef: float,
    gamma: float,
) -> list[torch.Tensor]:
    """
    Calculates discounted returns for REINFORCE++ (https://arxiv.org/pdf/2501.03262)

    Args:
        rewards (Tensor): A tensor of scalar rewards for each sequence.
        kl (List[Tensor]): List of per-token KL divergence tensors for sequence chunks.
        loss_masks (List[Tensor]): List of response-only loss masks for each full sequence.
        response_lengths (List[int]): The full length of each response sequence.
        total_lengths (List[int]): The full length of each sequence (prompt + response).
        kl_coef (float): Coefficient for the KL penalty.
        gamma (float): The discount factor.

    Returns:
        List[torch.Tensor]: A list of return (G_t) tensors for the
                            local sequence chunks owned by the current GPU rank.
    """
    try:
        from megatron.core import mpu

        cp_size = mpu.get_context_parallel_world_size()
    except ModuleNotFoundError:
        cp_size = 1

    final_returns_chunks = []
    for i in range(len(rewards)):
        local_kl_chunk = kl[i]
        total_len, response_len = total_lengths[i], response_lengths[i]

        if cp_size > 1:
            # Step 1,2:Gather all chunks and token_offsets from all ranks and reconstruct the full response tensor by splitting and placing each part
            from slime.backends.megatron_utils.cp_utils import all_gather_with_cp

            full_kl_response = all_gather_with_cp(local_kl_chunk, total_len, response_len)
        else:
            full_kl_response = local_kl_chunk

        # Step 3: Compute returns on full response kl tensor.
        full_mask = loss_masks[i]
        assert full_mask.sum().item() > 0, f"Sequence at index {i} is fully masked."
        masked_kl = full_kl_response * full_mask
        token_level_rewards = -kl_coef * masked_kl
        last_idx = full_mask.nonzero(as_tuple=True)[0][-1]
        token_level_rewards[last_idx] += rewards[i]

        returns_for_seq = torch.zeros_like(token_level_rewards)
        running_return = 0.0
        for t in reversed(range(token_level_rewards.size(0))):
            # G_t = r_t + gamma * G_{t+1}
            running_return = token_level_rewards[t] + gamma * running_return
            returns_for_seq[t] = running_return

        # Step 4: Pick up the results corresponding to our local chunk's parts.
        if cp_size > 1:
            from slime.backends.megatron_utils.cp_utils import slice_log_prob_with_cp

            local_returns_chunk = slice_log_prob_with_cp(returns_for_seq, total_len, response_len)
        else:
            local_returns_chunk = returns_for_seq

        final_returns_chunks.append(local_returns_chunk)

    return final_returns_chunks


def get_reinforce_plus_plus_baseline_advantages(
    rewards: torch.Tensor,
    kl: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
    kl_coef: float,
) -> list[torch.Tensor]:
    """
    Calculates the unwhitened advantages for the REINFORCE++-baseline algorithm.
    Broadcasting the scalar (reward - group_baseline) to each token.

    Args:
        rewards (Tensor): A tensor of scalar rewards, where the group-wise
                                baseline has already been subtracted.
        kl (list[Tensor]): A list of per-token KL divergence tensors. Used to
                                 get the shape for broadcasting.
        loss_masks (list[Tensor]): A list of per-token loss masks.
        kl_coef (float): Coefficient for the KL penalty.

    Returns:
        list[Tensor]: A list of tensors containing the unwhitened advantages.
    """
    # Broadcast to get unwhitened advantages
    unwhitened_advantages = [
        torch.ones_like(kl_tensor) * reward_val - kl_coef * kl_tensor
        for kl_tensor, reward_val in zip(kl, rewards, strict=False)
    ]

    return unwhitened_advantages


def _merge_anchor_token_ids(token_id: int | None, token_ids: list[int] | None) -> list[int]:
    merged: list[int] = []
    if token_ids is not None:
        merged.extend(int(x) for x in token_ids)
    if token_id is not None:
        merged.append(int(token_id))

    deduped: list[int] = []
    seen: set[int] = set()
    for token in merged:
        if token not in seen:
            seen.add(token)
            deduped.append(token)
    return deduped


def build_token_anchor_mask(
    response_tokens: torch.Tensor,
    *,
    token_id: int | None = None,
    token_ids: list[int] | None = None,
    token_interval: int | None = None,
    force_first_token: bool = True,
) -> torch.Tensor:
    """Build a boolean anchor mask over response tokens.

    Priority:
    1) token_interval (>0) if provided.
    2) token_id/token_ids exact-id matching.
    3) fallback to all-False.

    The first response position is forced as anchor by default to guarantee at
    least one anchor for non-empty responses.
    """
    assert response_tokens.ndim == 1, f"response_tokens must be 1D, got {tuple(response_tokens.shape)}"

    response_len = response_tokens.size(0)
    anchor_mask = torch.zeros(response_len, dtype=torch.bool, device=response_tokens.device)
    if response_len == 0:
        return anchor_mask

    if token_interval is not None and token_interval > 0:
        anchor_mask[::token_interval] = True
    else:
        merged_ids = _merge_anchor_token_ids(token_id, token_ids)
        if merged_ids:
            ids = torch.tensor(merged_ids, dtype=response_tokens.dtype, device=response_tokens.device)
            anchor_mask = (response_tokens.unsqueeze(-1) == ids.unsqueeze(0)).any(dim=-1)

    if force_first_token:
        anchor_mask[0] = True
    return anchor_mask


def _normalize_anchor_ratios(ratios: list[float] | tuple[float, ...] | float | None, *, name: str) -> list[float]:
    if ratios is None:
        return []
    if isinstance(ratios, (int, float)):
        ratios = [float(ratios)]
    else:
        ratios = [float(ratio) for ratio in ratios]
    for ratio in ratios:
        if not 0.0 <= ratio <= 1.0:
            raise ValueError(f"{name} must be in [0, 1], got {ratio}")
    return ratios


def build_ratio_anchor_mask(
    response_len: int,
    ratios: list[float],
    *,
    device: torch.device,
    valid_mask: torch.Tensor | None = None,
    avoid_tail_for_nonterminal: bool = True,
) -> torch.Tensor:
    anchor_mask = torch.zeros(response_len, dtype=torch.bool, device=device)
    if response_len == 0 or not ratios:
        return anchor_mask
    if valid_mask is None:
        valid_positions = torch.arange(response_len, device=device)
    else:
        if valid_mask.numel() != response_len:
            raise ValueError(f"valid_mask length {valid_mask.numel()} != response_len {response_len}")
        valid_positions = torch.nonzero(valid_mask.to(device=device, dtype=torch.bool), as_tuple=False).squeeze(-1)
    if valid_positions.numel() == 0:
        return anchor_mask

    last_valid_idx = valid_positions.numel() - 1
    for ratio in ratios:
        ratio_idx = int(round(last_valid_idx * ratio))
        ratio_idx = max(0, min(last_valid_idx, ratio_idx))
        if avoid_tail_for_nonterminal and ratio < 1.0 and last_valid_idx > 0:
            ratio_idx = min(ratio_idx, last_valid_idx - 1)
        anchor_mask[valid_positions[ratio_idx]] = True
    return anchor_mask


def build_critic_value_loss_masks_by_tokens(
    args: Namespace,
    tokens_list: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
) -> list[torch.Tensor]:
    """Build critic-only sparse value-loss masks without mutating policy loss masks."""
    if not getattr(args, "critic_token_loss", False):
        return loss_masks
    if len(tokens_list) != len(loss_masks):
        raise ValueError(f"tokens/loss_masks length mismatch: {len(tokens_list)} vs {len(loss_masks)}")

    critic_token_ratios = _normalize_anchor_ratios(
        getattr(args, "critic_token_ratios", None),
        name="critic_token_ratios",
    )
    critic_extra_tail_ratios = _normalize_anchor_ratios(
        getattr(args, "critic_extra_tail_ratio", None),
        name="critic_extra_tail_ratio",
    )
    critic_extra_tail_min_response_len = int(getattr(args, "critic_extra_tail_min_response_len", 0) or 0)
    if critic_extra_tail_min_response_len < 0:
        raise ValueError(
            "critic_extra_tail_min_response_len must be non-negative, "
            f"got {critic_extra_tail_min_response_len}"
        )

    critic_token_interval = getattr(args, "critic_token_interval", None)
    critic_token_ids = _merge_anchor_token_ids(
        getattr(args, "critic_token", None),
        getattr(args, "critic_token_ids", None),
    )
    has_ratios = len(critic_token_ratios) > 0
    has_extra_tail = len(critic_extra_tail_ratios) > 0
    has_interval = critic_token_interval is not None and int(critic_token_interval) > 0
    has_token_ids = bool(critic_token_ids)
    if not has_ratios and not has_interval and not has_token_ids and not has_extra_tail:
        return loss_masks

    sparse_loss_masks: list[torch.Tensor] = []
    for tokens, loss_mask in zip(tokens_list, loss_masks, strict=False):
        response_len = loss_mask.numel()
        if response_len == 0:
            sparse_loss_masks.append(loss_mask.clone())
            continue
        if tokens.numel() < response_len:
            raise ValueError(f"tokens length {tokens.numel()} is shorter than response loss mask length {response_len}")

        valid_positions = torch.nonzero(loss_mask.to(dtype=torch.bool), as_tuple=False).squeeze(-1)
        sparse_mask = torch.zeros_like(loss_mask, dtype=torch.bool)
        if valid_positions.numel() == 0:
            sparse_loss_masks.append(loss_mask * sparse_mask.to(dtype=loss_mask.dtype))
            continue

        if has_ratios:
            sparse_mask = build_ratio_anchor_mask(
                response_len,
                critic_token_ratios,
                device=loss_mask.device,
                valid_mask=loss_mask,
                avoid_tail_for_nonterminal=True,
            )
        elif has_interval:
            sparse_mask[valid_positions[:: int(critic_token_interval)]] = True
        elif has_token_ids:
            response_tokens = tokens[-response_len:].to(device=loss_mask.device)
            ids = torch.tensor(critic_token_ids, dtype=response_tokens.dtype, device=response_tokens.device)
            token_match_mask = (response_tokens.unsqueeze(-1) == ids.unsqueeze(0)).any(dim=-1)
            sparse_mask = token_match_mask.to(device=loss_mask.device) & loss_mask.to(dtype=torch.bool)

        if not has_ratios and (has_interval or has_token_ids):
            sparse_mask[valid_positions[0]] = True
            sparse_mask[valid_positions[-1]] = True
        if has_extra_tail and valid_positions.numel() >= critic_extra_tail_min_response_len:
            sparse_mask |= build_ratio_anchor_mask(
                response_len,
                critic_extra_tail_ratios,
                device=loss_mask.device,
                valid_mask=loss_mask,
                avoid_tail_for_nonterminal=False,
            )
        sparse_loss_masks.append(loss_mask * sparse_mask.to(dtype=loss_mask.dtype))

    return sparse_loss_masks


def broadcast_values_with_anchor_mask(
    values: torch.Tensor,
    anchor_mask: torch.Tensor,
    *,
    force_first_anchor: bool = True,
) -> torch.Tensor:
    """Broadcast anchor values forward to all following tokens."""
    assert values.ndim == 1, f"values must be 1D, got {tuple(values.shape)}"
    assert anchor_mask.ndim == 1, f"anchor_mask must be 1D, got {tuple(anchor_mask.shape)}"
    assert values.size(0) == anchor_mask.size(0), f"shape mismatch: values={values.shape}, mask={anchor_mask.shape}"

    if values.numel() == 0:
        return values

    if anchor_mask.dtype != torch.bool:
        anchor_mask = anchor_mask.to(dtype=torch.bool)

    if not anchor_mask.any():
        anchor_mask = anchor_mask.clone()
        anchor_mask[0] = True
    elif force_first_anchor and not anchor_mask[0]:
        anchor_mask = anchor_mask.clone()
        anchor_mask[0] = True

    anchor_positions = torch.nonzero(anchor_mask, as_tuple=False).squeeze(-1)
    anchor_values = values.index_select(0, anchor_positions)
    positions = torch.arange(values.size(0), device=values.device)
    anchor_group = torch.searchsorted(anchor_positions, positions, right=True) - 1
    anchor_group = anchor_group.clamp_min(0)
    return anchor_values.index_select(0, anchor_group)


def broadcast_policy_values_by_tokens(
    args: Namespace,
    values_list: list[torch.Tensor],
    tokens_list: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
    loss_masks: list[torch.Tensor] | None = None,
) -> list[torch.Tensor]:
    """Apply token-anchor value broadcasting for PPO value traces.

    Supports both CP and non-CP: with CP enabled, values are gathered to full
    response length, broadcasted there, then sliced back to local chunks.
    """
    if len(values_list) != len(tokens_list):
        raise ValueError(f"values/tokens length mismatch: {len(values_list)} vs {len(tokens_list)}")

    if len(values_list) != len(total_lengths) or len(values_list) != len(response_lengths):
        raise ValueError(
            f"batch length mismatch: values={len(values_list)}, total_lengths={len(total_lengths)}, response_lengths={len(response_lengths)}"
        )

    try:
        from megatron.core import mpu

        cp_size = mpu.get_context_parallel_world_size()
    except ModuleNotFoundError:
        cp_size = 1
    all_gather_with_cp = None
    slice_log_prob_with_cp = None
    if cp_size > 1:
        from slime.backends.megatron_utils.cp_utils import all_gather_with_cp, slice_log_prob_with_cp

    policy_token = getattr(args, "policy_token", None)
    policy_token_ids = getattr(args, "policy_token_ids", None)
    policy_token_interval = getattr(args, "policy_token_interval", None)
    policy_token_ratios = _normalize_anchor_ratios(
        getattr(args, "policy_token_ratios", None),
        name="policy_token_ratios",
    )

    out_values: list[torch.Tensor] = []
    if loss_masks is not None and len(loss_masks) != len(values_list):
        raise ValueError(f"loss_masks/values length mismatch: {len(loss_masks)} vs {len(values_list)}")

    for sample_idx, (values, tokens, total_len, response_len) in enumerate(zip(
        values_list,
        tokens_list,
        total_lengths,
        response_lengths,
        strict=False,
    )):
        if response_len == 0:
            out_values.append(values)
            continue

        if values.size(0) == response_len:
            full_values = values
            need_slice_back = False
        elif cp_size > 1:
            full_values = all_gather_with_cp(values, total_len, response_len)
            need_slice_back = True
        else:
            raise ValueError(
                f"Expected values length == response_len ({response_len}) in non-CP mode, got {values.size(0)}"
            )

        if torch.is_tensor(tokens):
            response_tokens = tokens[-response_len:]
            if response_tokens.device != full_values.device:
                response_tokens = response_tokens.to(device=full_values.device)
        else:
            response_tokens = torch.tensor(tokens[-response_len:], dtype=torch.long, device=full_values.device)
        if policy_token_ratios:
            valid_mask = None
            if loss_masks is not None:
                valid_mask = loss_masks[sample_idx].to(device=full_values.device)
            anchor_mask = build_ratio_anchor_mask(
                int(response_len),
                policy_token_ratios,
                device=full_values.device,
                valid_mask=valid_mask,
                avoid_tail_for_nonterminal=True,
            )
            force_first_anchor = False
        else:
            anchor_mask = build_token_anchor_mask(
                response_tokens,
                token_id=policy_token,
                token_ids=policy_token_ids,
                token_interval=policy_token_interval,
                force_first_token=True,
            )
            force_first_anchor = True

        broadcast_full_values = broadcast_values_with_anchor_mask(
            full_values,
            anchor_mask,
            force_first_anchor=force_first_anchor,
        )
        if need_slice_back:
            broadcast_local_values = slice_log_prob_with_cp(broadcast_full_values, total_len, response_len)
            out_values.append(broadcast_local_values)
        else:
            out_values.append(broadcast_full_values)

    return out_values


def get_advantages_and_returns(
    total_len: int,
    response_len: int,
    values: torch.Tensor,
    rewards: torch.Tensor,
    gamma: float,
    lambd: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Function that computes advantages and returns from rewards and values.
    Calculated as in the original PPO paper: https://arxiv.org/abs/1707.06347
    Note that rewards may include a KL divergence loss term.

    Advantages looks like this:
    Adv1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
            - V1 + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

    Returns looks like this:
    Ret1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
                + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

    Input:
    - values: Tensor of shape (response_size,)
    - rewards: Tensor of shape (response_size,)

    Output:
    - advantages: Tensor of shape (response_size,)
    - returns: Tensor of shape (response_size,)
    """
    from megatron.core import mpu

    cp_size = mpu.get_context_parallel_world_size()
    if cp_size > 1:
        from slime.backends.megatron_utils.cp_utils import all_gather_with_cp

        full_rewards = all_gather_with_cp(rewards, total_len, response_len)
        full_values = all_gather_with_cp(values, total_len, response_len)
    else:
        full_rewards = rewards
        full_values = values

    lastgaelam = 0
    advantages_reversed = []

    for t in reversed(range(response_len)):
        nextvalues = full_values[t + 1] if t < response_len - 1 else 0.0
        delta = full_rewards[t] + gamma * nextvalues - full_values[t]
        lastgaelam = delta + gamma * lambd * lastgaelam
        advantages_reversed.append(lastgaelam)
    full_advantages = torch.tensor(advantages_reversed[::-1], dtype=full_values.dtype, device=full_values.device)
    full_returns = full_advantages + full_values

    if cp_size > 1:
        from slime.backends.megatron_utils.cp_utils import slice_log_prob_with_cp

        advantages = slice_log_prob_with_cp(full_advantages, total_len, response_len)
        returns = slice_log_prob_with_cp(full_returns, total_len, response_len)
    else:
        advantages = full_advantages
        returns = full_returns

    return advantages.detach(), returns


def get_advantages_and_returns_batch(
    total_lengths,
    response_lengths,
    values_list,
    rewards_list,
    gamma,
    lambd,
    chunked: bool = True,
    policy_lambd=None,
):
    """
    Batched GAE with CP support.
    Input:
        total_lengths:     list[int], each sample's total_len
        response_lengths:  list[int], each sample's response_len
        values_list:       list[Tensor], each shape = [resp_len_i]
        rewards_list:      list[Tensor], same shape
        policy_lambd:      optional scalar or per-response lambdas for actor advantages;
                           critic returns continue to use lambd
    Output:
        advantages_list:   list[Tensor], each shape = [resp_len_i]
        returns_list:      list[Tensor], same shape
    """

    from megatron.core import mpu

    with torch.no_grad():
        B = len(response_lengths)
        assert B == len(values_list)
        assert B == len(rewards_list)

        cp_size = mpu.get_context_parallel_world_size()
        device = values_list[0].device
        dtype = values_list[0].dtype

        if cp_size > 1:
            from slime.backends.megatron_utils.cp_utils import all_gather_with_cp

            full_values_list = []
            full_rewards_list = []

            for total_len, resp_len, v, r in zip(
                total_lengths, response_lengths, values_list, rewards_list, strict=False
            ):
                full_v = all_gather_with_cp(v, total_len, resp_len)
                full_r = all_gather_with_cp(r, total_len, resp_len)
                full_values_list.append(full_v)
                full_rewards_list.append(full_r)

            # full_values_list[i].shape = [total_len_i]
        else:
            full_values_list = values_list
            full_rewards_list = rewards_list

        # pad to max_len for batched GAE
        max_len = max(response_lengths)

        full_values = torch.zeros(B, max_len, device=device, dtype=dtype)
        full_rewards = torch.zeros(B, max_len, device=device, dtype=dtype)

        for i in range(B):
            L = response_lengths[i]
            full_values[i, :L] = full_values_list[i][:L]
            full_rewards[i, :L] = full_rewards_list[i][:L]

        if not chunked:
            full_advantages, full_returns = vanilla_gae(
                rewards=full_rewards,
                values=full_values,
                gamma=gamma,
                lambd=lambd,
            )
        else:
            full_advantages, full_returns = chunked_gae(
                rewards=full_rewards,
                values=full_values,
                gamma=gamma,
                lambd=lambd,
            )

        # Critic returns above retain their own lambda. Only replace the actor
        # advantages, reusing the gathered/padded tensors for context parallelism.
        if policy_lambd is not None:
            if isinstance(policy_lambd, (list, tuple)):
                if len(policy_lambd) != B:
                    raise ValueError("policy_lambd must contain one lambda per response.")
                policy_lambd = torch.tensor(policy_lambd, device=device, dtype=dtype)
            gae = chunked_gae if chunked else vanilla_gae
            full_advantages, _ = gae(full_rewards, full_values, gamma, policy_lambd)

        advantages_list = []
        returns_list = []

        if cp_size > 1:
            from slime.backends.megatron_utils.cp_utils import slice_log_prob_with_cp

            for total_len, resp_len, adv_row, ret_row in zip(
                total_lengths,
                response_lengths,
                full_advantages,
                full_returns,
                strict=False,
            ):
                adv_full = adv_row  # shape = [resp_len_i padded to max_len]
                ret_full = ret_row

                adv_sliced = slice_log_prob_with_cp(adv_full[:resp_len], total_len, resp_len)
                ret_sliced = slice_log_prob_with_cp(ret_full[:resp_len], total_len, resp_len)

                advantages_list.append(adv_sliced)
                returns_list.append(ret_sliced)

        else:
            for i in range(B):
                L = response_lengths[i]
                advantages_list.append(full_advantages[i, :L])
                returns_list.append(full_returns[i, :L])

    return advantages_list, returns_list


def get_segmented_value_difference_advantages_batch(
    total_lengths: list[int],
    response_lengths: list[int],
    values_list: list[torch.Tensor],
    rewards_list: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
    ratios: list[float],
    gamma: float,
) -> list[torch.Tensor]:
    """Compute one shared one-step advantage for each length-based segment.

    Ratio-selected response positions define segment starts; position zero is
    always included. For a segment [b_i, b_{i+1}), every valid token receives

        sum_j gamma^j r_{b_i+j} + gamma^len V(b_{i+1}) - V(b_i).

    The last segment bootstraps from zero. With gamma=1 and terminal-only
    reward, this is V(b_{i+1}) - V(b_i) for intermediate segments and
    R - V(b_i) for the last segment.
    """
    from megatron.core import mpu

    if not (
        len(total_lengths)
        == len(response_lengths)
        == len(values_list)
        == len(rewards_list)
        == len(loss_masks)
    ):
        raise ValueError("Segmented advantage inputs must have the same batch size.")

    normalized_ratios = sorted(set(float(ratio) for ratio in ratios))
    for ratio in normalized_ratios:
        if not 0.0 <= ratio <= 1.0:
            raise ValueError(f"Segment ratio must be in [0, 1], got {ratio}.")

    cp_size = mpu.get_context_parallel_world_size()
    if cp_size > 1:
        from slime.backends.megatron_utils.cp_utils import all_gather_with_cp, slice_log_prob_with_cp
    advantages_list: list[torch.Tensor] = []

    with torch.no_grad():
        for total_len, response_len, values, rewards, loss_mask in zip(
            total_lengths,
            response_lengths,
            values_list,
            rewards_list,
            loss_masks,
            strict=False,
        ):
            if cp_size > 1:
                full_values = all_gather_with_cp(values, total_len, response_len)
                full_rewards = all_gather_with_cp(rewards, total_len, response_len)
            else:
                full_values = values
                full_rewards = rewards

            if full_values.numel() != response_len or full_rewards.numel() != response_len:
                raise ValueError(
                    "Expected full response-aligned values and rewards, got "
                    f"values={full_values.numel()}, rewards={full_rewards.numel()}, "
                    f"response_len={response_len}."
                )
            if loss_mask.numel() != response_len:
                raise ValueError(
                    f"loss_mask length {loss_mask.numel()} != response_len {response_len}."
                )

            full_advantages = torch.zeros_like(full_values)
            valid_positions = torch.nonzero(
                loss_mask.to(device=full_values.device, dtype=torch.bool),
                as_tuple=False,
            ).squeeze(-1)
            num_valid = int(valid_positions.numel())
            if num_valid > 0:
                boundary_ordinals = {0}
                for ratio in normalized_ratios:
                    boundary_ordinal = int(round(num_valid * ratio))
                    boundary_ordinals.add(min(boundary_ordinal, num_valid - 1))
                boundary_ordinals = sorted(boundary_ordinals)

                for segment_idx, start_ordinal in enumerate(boundary_ordinals):
                    end_ordinal = (
                        boundary_ordinals[segment_idx + 1]
                        if segment_idx + 1 < len(boundary_ordinals)
                        else num_valid
                    )
                    if end_ordinal <= start_ordinal:
                        continue

                    segment_positions = valid_positions[start_ordinal:end_ordinal]
                    segment_rewards = full_rewards.index_select(0, segment_positions)
                    if gamma == 1.0:
                        discounted_reward = segment_rewards.sum()
                    else:
                        discounts = gamma ** torch.arange(
                            segment_positions.numel(),
                            device=full_values.device,
                            dtype=full_values.dtype,
                        )
                        discounted_reward = (segment_rewards * discounts).sum()

                    start_position = valid_positions[start_ordinal]
                    if end_ordinal < num_valid:
                        next_position = valid_positions[end_ordinal]
                        bootstrap = (gamma ** segment_positions.numel()) * full_values[next_position]
                    else:
                        bootstrap = full_values.new_zeros(())

                    segment_advantage = discounted_reward + bootstrap - full_values[start_position]
                    full_advantages[segment_positions] = segment_advantage

            if cp_size > 1:
                advantages = slice_log_prob_with_cp(full_advantages, total_len, response_len)
            else:
                advantages = full_advantages
            advantages_list.append(advantages.detach())

    return advantages_list


def vanilla_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    gamma: float,
    lambd: float,
):
    B, T = rewards.shape
    device = rewards.device
    dtype = rewards.dtype

    lastgaelam = torch.zeros(B, device=device, dtype=dtype)
    adv_rev = []

    for t in reversed(range(T)):
        next_value = values[:, t + 1] if t < T - 1 else 0.0
        delta = rewards[:, t] + gamma * next_value - values[:, t]
        lastgaelam = delta + gamma * lambd * lastgaelam
        adv_rev.append(lastgaelam)

    full_advantages = torch.stack(adv_rev[::-1], dim=1)  # [B, max_len]
    full_returns = full_advantages + values  # [B, max_len]
    return full_advantages, full_returns


def chunked_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    gamma: float,
    lambd: float,
    chunk_size: int = 128,
):
    """
    Compute Generalized Advantage Estimation (GAE) using a FlashLinearAttention-
    inspired algorithm: parallel prefix scan within chunks and recurrent state
    propagation across chunks.

    This reduces the sequential dependency length from O(T) to O(T / chunk_size),
    while keeping chunk computations fully parallelizable (O(C^2) per chunk).

    Args:
        rewards (Tensor): [B, T] reward sequence.
        values (Tensor):  [B, T] value predictions. The next-value of the final
                          step is assumed to be zero (standard PPO convention).
        gamma (float): discount factor.
        lambd (float or Tensor): Shared GAE lambda or one lambda per response, shape [B].
        chunk_size (int): sequence chunk length for parallel scan.

    Returns:
        advantages (Tensor): [B, T] computed advantages.
        returns (Tensor):    [B, T] advantages + values.
    """

    # -------------------------------------------------------------------------
    # Validate inputs
    # -------------------------------------------------------------------------
    assert rewards.ndim == 2 and values.ndim == 2
    B, T = rewards.shape
    assert values.shape == (B, T)

    device = rewards.device
    dtype = rewards.dtype

    # -------------------------------------------------------------------------
    # Build δ_t = r_t + γ * V_{t+1} - V_t   with V_{T} = 0
    # -------------------------------------------------------------------------
    next_values = torch.cat(
        [values[:, 1:], torch.zeros(B, 1, device=device, dtype=dtype)],
        dim=1,
    )
    deltas = rewards + gamma * next_values - values

    # Reformulate backward GAE as a forward scan on the reversed sequence:
    #   S[i] = Δ[i] + w * S[i - 1],   w = γλ
    w = gamma * lambd
    deltas_rev = torch.flip(deltas, dims=[1])  # [B, T]

    # -------------------------------------------------------------------------
    # Pad to a multiple of chunk_size
    # -------------------------------------------------------------------------
    if T % chunk_size != 0:
        pad = chunk_size - (T % chunk_size)
        deltas_rev = F.pad(deltas_rev, (0, pad))
    else:
        pad = 0

    B, T_pad = deltas_rev.shape
    n_chunks = T_pad // chunk_size

    deltas_chunks = deltas_rev.view(B, n_chunks, chunk_size)

    # -------------------------------------------------------------------------
    # Construct the intra-chunk parallel scan kernel M
    #
    # For a chunk Δ[0..C-1], we want:
    #   S_local[t] = sum_{k=0..t} w^(t-k) * Δ[k]
    #
    # This is implemented as:
    #   S_local = Δ @ M
    #
    # where:
    #   M[i, j] = w^(j - i)    if j >= i
    #             0            otherwise
    # -------------------------------------------------------------------------
    idx = torch.arange(chunk_size, device=device)
    row = idx[:, None]
    col = idx[None, :]
    diff = col - row

    M = torch.zeros(chunk_size, chunk_size, device=device, dtype=dtype)
    mask = diff >= 0

    per_response_lambda = isinstance(w, torch.Tensor) and w.ndim == 1
    if per_response_lambda:
        if w.shape != (B,):
            raise ValueError("Per-response GAE lambda must have shape [batch_size].")
        M = w[:, None, None] ** diff.clamp_min(0).to(dtype)[None, :, :]
        M = M * mask[None, :, :]
    elif w == 0.0:
        M[mask & (diff == 0)] = 1.0
    else:
        M[mask] = w ** diff[mask].to(dtype)

    # pow_vec[t] = w^(t+1), used to inject the recurrent state s_prev
    if per_response_lambda:
        pow_vec = w[:, None] ** torch.arange(1, chunk_size + 1, device=device, dtype=dtype)[None, :]
    elif w == 0.0:
        pow_vec = torch.zeros(chunk_size, device=device, dtype=dtype)
    else:
        pow_vec = w ** torch.arange(1, chunk_size + 1, device=device, dtype=dtype)

    # -------------------------------------------------------------------------
    # Parallel compute local chunk results (assuming initial state = 0)
    # -------------------------------------------------------------------------
    if per_response_lambda:
        S_local_chunks = torch.bmm(deltas_chunks, M)
    else:
        deltas_flat = deltas_chunks.reshape(B * n_chunks, chunk_size)
        S_local_flat = deltas_flat @ M
        S_local_chunks = S_local_flat.view(B, n_chunks, chunk_size)

    # Effective length of each chunk (the last chunk may be padded)
    lengths = [chunk_size] * n_chunks
    if pad > 0:
        lengths[-1] = chunk_size - pad

    # -------------------------------------------------------------------------
    # Recurrent propagation between chunks
    #
    # Each chunk contributes:
    #   S_global[t] = S_local[t] + w^(t+1) * s_prev
    #
    # And updates:
    #   s_prev = S_global[last_t]
    # -------------------------------------------------------------------------
    S_rev = deltas_rev.new_zeros(B, T_pad)
    s_prev = torch.zeros(B, device=device, dtype=dtype)

    for c in range(n_chunks):
        Lc = lengths[c]
        start = c * chunk_size
        end = start + Lc

        S_local = S_local_chunks[:, c, :Lc]
        powers = pow_vec[:, :Lc] if per_response_lambda else pow_vec[:Lc]
        S_global = S_local + s_prev.unsqueeze(1) * powers

        S_rev[:, start:end] = S_global
        s_prev = S_global[:, -1]  # state for next chunk

    # Remove padding and flip back to original time order
    if pad > 0:
        S_rev = S_rev[:, :T]

    advantages = torch.flip(S_rev, dims=[1])
    returns = advantages + values

    return advantages, returns


def calculate_log_probs_and_entropy(logits, tokens, tp_group, with_entropy: bool = False, chunk_size: int = -1):
    logits = logits.contiguous()
    # TODO: not sure why we need to clone the logits here.
    # Without the clone, the backward will trigger inplace edit error.
    # It seems that the function with tp will modify the logits inplace.
    entropy = None
    if logits.size(0) != 0:
        if chunk_size > 0:
            num_chunks = (logits.size(0) - 1) // chunk_size + 1
            tokens_chunks = tokens.chunk(num_chunks, dim=0)
            logits_chunks = logits.chunk(num_chunks, dim=0)
            log_probs = []
            for tokens_chunk, logits_chunk in zip(tokens_chunks, logits_chunks, strict=True):
                log_prob = compute_log_probs(logits_chunk.clone(), tokens_chunk, tp_group)
                log_probs.append(log_prob)
            log_prob = torch.cat(log_probs, dim=0)
            if with_entropy:
                entropys = []
                for _, logits_chunk in zip(tokens_chunks, logits_chunks, strict=True):
                    entropy = compute_entropy_from_logits(logits_chunk.clone(), tp_group)
                    entropys.append(entropy)
                entropy = torch.cat(entropys, dim=0)
        else:
            log_prob = compute_log_probs(logits.clone(), tokens, tp_group)
            if with_entropy:
                entropy = compute_entropy_from_logits(logits.clone(), tp_group)
    else:
        log_prob = logits.new_zeros((0,))
        if with_entropy:
            entropy = logits.new_zeros((0,))

    return log_prob, entropy
