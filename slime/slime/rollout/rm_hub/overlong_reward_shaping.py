import torch

from slime.utils.types import Sample


def _get_overlong_config(args):
    max_len = args.rollout_max_response_len
    buffer_len = args.overlong_buffer_len
    expected_len = args.overlong_expected_len

    if buffer_len <= 0:
        raise ValueError("--overlong-buffer-len must be positive when using overlong reward shaping.")

    if expected_len is None:
        expected_len = max_len - buffer_len

    if expected_len < 0:
        raise ValueError(
            f"overlong expected length must be non-negative, got {expected_len}. "
            "Increase --rollout-max-response-len or decrease --overlong-buffer-len."
        )

    if expected_len > max_len:
        raise ValueError(
            f"overlong expected length ({expected_len}) must be <= rollout max response length ({max_len})."
        )

    return expected_len, max_len, buffer_len, args.overlong_penalty_factor


def _overlong_penalty(response_length: int, expected_len: int, max_len: int, buffer_len: int, penalty_factor: float):
    if response_length <= expected_len:
        return 0.0

    overlong_ratio = min((response_length - expected_len) / buffer_len, 1.0)
    if response_length >= max_len:
        overlong_ratio = 1.0

    return -penalty_factor * overlong_ratio


def _normalize_rewards_like_default(args, rewards: list[float]):
    if (
        args.advantage_estimator in ["grpo", "gspo", "reinforce_plus_plus_baseline"]
        and args.rewards_normalization
    ):
        reward_tensor = torch.tensor(rewards, dtype=torch.float)
        if reward_tensor.shape[-1] == args.n_samples_per_prompt * args.rollout_batch_size:
            reward_tensor = reward_tensor.reshape(-1, args.n_samples_per_prompt)
        else:
            reward_tensor = reward_tensor.view(-1, reward_tensor.shape[-1])

        reward_tensor = reward_tensor - reward_tensor.mean(dim=-1, keepdim=True)

        if args.advantage_estimator in ["grpo", "gspo"] and args.grpo_std_normalization:
            reward_tensor = reward_tensor / (reward_tensor.std(dim=-1, keepdim=True) + 1e-6)

        return reward_tensor.flatten().tolist()

    return rewards


def post_process_rewards(args, samples: list[Sample], **kwargs):
    """Apply DAPO-style soft overlong punishment before advantage computation.

    The dynamic sampling filter has already run by this point, so this function
    shapes the training rewards without changing prompt-group selection.
    """

    expected_len, max_len, buffer_len, penalty_factor = _get_overlong_config(args)

    raw_rewards = [float(sample.get_reward_value(args)) for sample in samples]
    shaped_rewards = []
    for sample, raw_reward in zip(samples, raw_rewards, strict=False):
        penalty = _overlong_penalty(
            sample.response_length,
            expected_len,
            max_len,
            buffer_len,
            penalty_factor,
        )
        shaped_rewards.append(raw_reward + penalty)

    rewards = _normalize_rewards_like_default(args, shaped_rewards)
    return raw_rewards, rewards
