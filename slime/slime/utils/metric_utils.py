import logging
import math
from collections.abc import Mapping
from typing import Any, Literal

import numpy as np
import torch

logger = logging.getLogger(__name__)


EXPLAINED_VARIANCE_STAT_NAMES = (
    "count",
    "target_sum",
    "target_square_sum",
    "residual_sum",
    "residual_square_sum",
)
VALUE_EXPLAINED_VARIANCE_STAT_PREFIX = "__value_explained_variance_"


def compute_masked_explained_variance_stats(
    targets: torch.Tensor,
    predictions: torch.Tensor,
    mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Return additive statistics for a masked explained-variance metric."""
    targets = targets.detach().float().flatten()
    predictions = predictions.detach().float().flatten()
    mask = mask.detach().float().flatten()
    if targets.shape != predictions.shape or targets.shape != mask.shape:
        raise ValueError(
            "Explained-variance targets, predictions, and mask must have the same shape, "
            f"got {tuple(targets.shape)}, {tuple(predictions.shape)}, and {tuple(mask.shape)}"
        )

    residuals = targets - predictions
    return {
        "count": mask.sum(),
        "target_sum": (targets * mask).sum(),
        "target_square_sum": (targets.square() * mask).sum(),
        "residual_sum": (residuals * mask).sum(),
        "residual_square_sum": (residuals.square() * mask).sum(),
    }


def compute_explained_variance_from_stats(
    stats: Mapping[str, float | torch.Tensor],
    epsilon: float = 1e-5,
) -> float:
    """Compute explained variance from globally aggregated additive statistics."""
    missing = [name for name in EXPLAINED_VARIANCE_STAT_NAMES if name not in stats]
    if missing:
        raise KeyError(f"Missing explained-variance statistics: {missing}")

    values = {
        name: float(value.detach().item()) if isinstance(value, torch.Tensor) else float(value)
        for name, value in stats.items()
        if name in EXPLAINED_VARIANCE_STAT_NAMES
    }
    count = values["count"]
    if count <= 0:
        return 0.0

    target_mean = values["target_sum"] / count
    residual_mean = values["residual_sum"] / count
    target_variance = max(values["target_square_sum"] / count - target_mean**2, 0.0)
    residual_variance = max(values["residual_square_sum"] / count - residual_mean**2, 0.0)
    if target_variance <= epsilon:
        return 0.0
    return 1.0 - residual_variance / (target_variance + epsilon)


def dict_add_prefix(d: dict[str, Any], prefix: str) -> dict[str, Any]:
    return {f"{prefix}{k}": v for k, v in d.items()}


def compute_pass_rate(
    flat_rewards: list[float],
    group_size: int,
    num_groups: int | None = None,
):
    if group_size == 1:
        return {}

    if num_groups is None:
        num_groups = len(flat_rewards) // group_size

    pass_rate_name_list = [2**i for i in range(int(math.log2(group_size)) + 1)]

    assert len(flat_rewards) == num_groups * group_size, f"{len(flat_rewards)=} {num_groups=} {group_size=}"
    rewards_of_group = np.array(flat_rewards).reshape(num_groups, group_size)

    log_dict = {}
    for k in pass_rate_name_list:
        num_correct = np.sum(rewards_of_group == 1, axis=1)
        num_samples = np.full(num_groups, group_size)

        pass_k_estimates = _estimate_pass_at_k(num_samples, num_correct, k)

        pass_k = np.mean(pass_k_estimates)
        log_dict[f"pass@{k}"] = pass_k

    return log_dict


def _estimate_pass_at_k(num_samples, num_correct, k):
    """
    Estimates pass@k of each problem and returns them in an array.
    """

    def estimator(n, c, k):
        """
        Calculates 1 - comb(n - c, k) / comb(n, k).
        """
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    return np.array([estimator(int(n), int(c), k) for n, c in zip(num_samples, num_correct, strict=False)])


def compute_statistics(values: list[float]) -> dict[str, float]:
    values = np.array(values)
    return {
        "mean": np.mean(values).item(),
        "median": np.median(values).item(),
        "max": np.max(values).item(),
        "min": np.min(values).item(),
    }


def compression_ratio(
    data: str | bytes,
    *,
    encoding: str = "utf-8",
    algorithm: Literal["zlib", "gzip", "bz2", "lzma"] = "zlib",
    level: int = 9,
) -> tuple[float, float]:
    if isinstance(data, str):
        raw = data.encode(encoding)
    else:
        raw = data

    original = len(raw)
    if original == 0:
        return float("inf"), 0.0

    if algorithm == "zlib":
        import zlib

        compressed = zlib.compress(raw, level)
    elif algorithm == "gzip":
        import gzip

        compressed = gzip.compress(raw, compresslevel=level)
    elif algorithm == "bz2":
        import bz2

        compressed = bz2.compress(raw, compresslevel=level)
    elif algorithm == "lzma":
        import lzma

        compressed = lzma.compress(raw, preset=level)
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    comp_len = len(compressed)
    if comp_len == 0:
        return float("inf"), 100.0

    ratio = original / comp_len
    savings_pct = 100.0 * (1.0 - comp_len / original)
    return ratio, savings_pct


def has_repetition(text: str):
    if len(text) > 10000 and compression_ratio(text[-10000:])[0] > 10:
        return True
    else:
        return False


def compute_rollout_step(args, rollout_id):
    if args.wandb_always_use_train_step:
        return rollout_id * args.rollout_batch_size * args.n_samples_per_prompt // args.global_batch_size
    return rollout_id
