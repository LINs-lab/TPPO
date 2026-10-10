"""Opt-in VAPO settings and normalization over one optimizer step."""

import math


def validate_vapo_args(args):
    decoupled = getattr(args, "use_decoupled_gae", False)
    adaptive = getattr(args, "use_adaptive_lambda", False)
    token_loss = getattr(args, "ppo_token_level_loss", False)
    positive_nll = getattr(args, "use_positive_nll_loss", False)
    if not any((decoupled, adaptive, token_loss, positive_nll)):
        return
    if args.advantage_estimator != "ppo" or args.train_backend != "megatron":
        raise ValueError("VAPO options require --advantage-estimator ppo and --train-backend megatron.")
    if adaptive and not decoupled:
        raise ValueError("--use-adaptive-lambda requires --use-decoupled-gae.")
    if decoupled:
        for name in ("lambd_actor", "lambd_critic"):
            value = getattr(args, name)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be finite and in [0, 1].")
        if getattr(args, "ppo_actor_segment_ratios", None) is not None:
            raise ValueError("Decoupled GAE cannot be combined with --ppo-actor-segment-ratios.")
    if adaptive and (not math.isfinite(args.alpha) or args.alpha <= 0):
        raise ValueError("--alpha must be finite and positive.")
    if token_loss and getattr(args, "custom_pg_loss_reducer_function_path", None):
        raise ValueError("--ppo-token-level-loss cannot be combined with a custom PG loss reducer.")
    if positive_nll:
        if not math.isfinite(args.positive_nll_coef) or args.positive_nll_coef < 0:
            raise ValueError("--positive-nll-coef must be finite and non-negative.")
        if not math.isfinite(args.positive_reward_threshold):
            raise ValueError("--positive-reward-threshold must be finite.")


def get_ppo_lambdas(args, response_lengths):
    """Return critic lambda and optional actor lambda(s); legacy defaults are unchanged."""
    if not getattr(args, "use_decoupled_gae", False):
        return args.lambd, None
    if not getattr(args, "use_adaptive_lambda", False):
        return args.lambd_critic, args.lambd_actor
    # Eq. (5) is negative for L < 1/alpha. Clamp to the valid GAE interval.
    actor_lambdas = [max(0.0, min(1.0, 1.0 - 1.0 / (args.alpha * max(length, 1))))
                     for length in response_lengths]
    return args.lambd_critic, actor_lambdas


def positive_example_masks(raw_rewards, loss_masks, threshold):
    """Select by the unnormalized verifier reward, never by advantage or shaped reward."""
    if raw_rewards is None or len(raw_rewards) != len(loss_masks):
        raise ValueError("Positive NLL requires one raw_reward per response.")
    return [mask * float(float(reward) > threshold)
            for reward, mask in zip(raw_rewards, loss_masks)]


def get_vapo_loss_normalizers(args, data_iterator, num_microbatches):
    """Count the actual next optimizer-step samples, across DP but not duplicate CP copies.

    Each micro-batch contributes an unnormalized token sum. Scaling it by
    outer/count cancels Megatron's later sample/token normalization, producing
    a global token mean independent of dynamic micro-batch packing.
    """
    import torch
    import torch.distributed as dist
    from megatron.core import mpu

    data = data_iterator.rollout_data
    offset = data_iterator.offset
    if data_iterator.micro_batch_indices is None:
        indices = list(range(offset, offset + num_microbatches * data_iterator.micro_batch_size))
    else:
        indices = [i for batch in data_iterator.micro_batch_indices[offset:offset + num_microbatches]
                   for i in batch]
    masks = [data["loss_masks"][i] for i in indices]
    counts = torch.stack([mask.sum() for mask in masks]).to(torch.float64)
    positive_tokens = counts.new_zeros(())
    if getattr(args, "use_positive_nll_loss", False):
        raw_rewards = data.get("positive_nll_rewards")
        if raw_rewards is None:
            raise ValueError("Positive NLL requires partition-aligned rollout_data['positive_nll_rewards'].")
        positive = counts.new_tensor([float(raw_rewards[i]) > args.positive_reward_threshold for i in indices])
        positive_tokens = (counts * positive).sum()
    totals = torch.stack((counts.sum(), positive_tokens, counts.clamp_min(1).sum()))
    if mpu.get_data_parallel_world_size(with_context_parallel=False) > 1:
        dist.all_reduce(totals, group=mpu.get_data_parallel_group(with_context_parallel=False))
    tokens, positive_tokens, megatron_tokens = totals.tolist()
    outer = (megatron_tokens if args.calculate_per_token_loss else
             data.get("dynamic_global_batch_size", args.global_batch_size))
    return {"tokens": tokens, "positive_tokens": positive_tokens, "outer": outer}


def normalize_vapo_token_sum(token_sum, normalizers, *, positive=False):
    denominator = normalizers["positive_tokens" if positive else "tokens"]
    return token_sum * (normalizers["outer"] / max(denominator, 1.0))
