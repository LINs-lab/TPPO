import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


def save_debug_train_data(args, *, rollout_id, rollout_data):
    if (path_template := args.save_debug_train_data) is not None:
        rank = torch.distributed.get_rank()
        path = Path(path_template.format(rollout_id=rollout_id, rank=rank))
        logger.info(f"Save debug train data to {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            dict(
                rollout_id=rollout_id,
                rank=rank,
                rollout_data=rollout_data,
            ),
            path,
        )


def _to_debug_cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    return value


def save_hlg_debug_value_distribution(args, *, rollout_id, rollout_data):
    num_samples = int(getattr(args, "hlg_debug_distribution_samples", 0))
    debug_entries = rollout_data.pop("hlg_value_distribution_debug", None)
    if num_samples <= 0 or not debug_entries:
        return

    samples = []
    scalar_keys = [
        "sample_indices",
        "raw_reward",
        "rewards",
        "response_lengths",
        "total_lengths",
        "truncated",
        "hlg_group_success_p",
        "hlg_group_num_correct",
        "hlg_group_size",
    ]
    tensor_keys = [
        "values",
        "value_std",
        "value_entropy",
        "value_p_success",
        "advantages",
        "returns",
    ]
    num_local_samples = len(rollout_data.get("response_lengths", []))
    for local_index, entry in enumerate(debug_entries):
        if entry is None:
            continue
        sample = dict(entry)
        sample["local_index"] = local_index
        for key in scalar_keys:
            values = rollout_data.get(key)
            if isinstance(values, (list, tuple)) and len(values) == num_local_samples and local_index < len(values):
                sample[key] = values[local_index]
        for key in tensor_keys:
            values = rollout_data.get(key)
            if isinstance(values, (list, tuple)) and local_index < len(values):
                sample[key] = _to_debug_cpu(values[local_index])
        samples.append(sample)
        if len(samples) >= num_samples:
            break

    if not samples:
        return

    output_dir = getattr(args, "hlg_debug_distribution_dir", None)
    if output_dir is None:
        logger.warning("Skip HLG value distribution debug save because hlg_debug_distribution_dir is not set.")
        return

    rank = torch.distributed.get_rank()
    path = Path(output_dir) / f"{rollout_id}_{rank}.pt"
    logger.info(f"Save sampled HLG value distribution debug data to {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "rollout_id": rollout_id,
            "rank": rank,
            "hlg_num_bins": int(getattr(args, "hlg_num_bins", 101)),
            "hlg_v_min": float(getattr(args, "hlg_v_min", 0.0)),
            "hlg_v_max": float(getattr(args, "hlg_v_max", 1.0)),
            "hlg_sigma_ratio": float(getattr(args, "hlg_sigma_ratio", 0.75)),
            "samples": samples,
        },
        path,
    )
