import logging
import os
import re
import json
from contextlib import contextmanager
from pathlib import Path

# TODO: may need to copy those 2 functions and do refactoring.
from megatron.training.checkpointing import load_checkpoint as _load_checkpoint_megatron
from megatron.training.checkpointing import save_checkpoint
from megatron.training.global_vars import get_args
import torch

from slime.utils import megatron_bridge_utils

try:
    # Here we patch out the `validate_non_overlapping_shards_metadata` in both functions
    # because it is really slow for large models with many shards.
    # TODO: find a less hacky way to do this.
    import torch.distributed as dist
    import torch.distributed._shard.sharding_spec as shard_spec
    from torch.distributed._shard.sharded_tensor import ShardedTensor
    from torch.distributed._shard.sharded_tensor.metadata import ShardedTensorMetadata
    from torch.distributed._shard.sharded_tensor.shard import Shard
    from torch.distributed._shard.sharded_tensor.utils import _parse_and_validate_remote_device
    from torch.distributed._shard.sharding_spec.api import EnumerableShardingSpec

    def __post_init__(self):
        pass

    EnumerableShardingSpec.__post_init__ = __post_init__

    @classmethod
    def _init_from_local_shards_and_global_metadata(  # type: ignore[override]
        cls,
        local_shards: list[Shard],
        sharded_tensor_metadata: ShardedTensorMetadata,
        process_group=None,
        init_rrefs=False,
        sharding_spec=None,
    ) -> ShardedTensor:
        """
        Initialize a ShardedTensor with local shards and a global
        ShardedTensorMetadata built on each rank.

        Warning: This API is experimental and subject to change. It does
                 not do cross rank validations, and fully rely on the user
                 for the correctness of sharded_tensor_metadata on each rank
        """
        process_group = cls._normalize_pg(process_group)
        current_rank = dist.get_rank()  # intentional to get global rank

        shards_metadata = sharded_tensor_metadata.shards_metadata

        local_shard_metadatas = []

        # collect local shard metadatas from the global sharded_tensor_metadata
        for shard_metadata in shards_metadata:  # type: ignore[attr-defined]
            rank, local_device = _parse_and_validate_remote_device(process_group, shard_metadata.placement)

            if current_rank == rank:
                local_shard_metadatas.append(shard_metadata)

        shards_metadata = sharded_tensor_metadata.shards_metadata
        tensor_properties = sharded_tensor_metadata.tensor_properties

        if sharding_spec is None:
            spec = shard_spec._infer_sharding_spec_from_shards_metadata(shards_metadata)
        else:
            spec = sharding_spec

        sharded_tensor = ShardedTensor.__new__(
            ShardedTensor,
            spec,
            sharded_tensor_metadata.size,
            dtype=tensor_properties.dtype,
            layout=tensor_properties.layout,
            pin_memory=tensor_properties.pin_memory,
            requires_grad=tensor_properties.requires_grad,
        )

        # done validation, add local_shards
        sharded_tensor._local_shards = local_shards
        sharded_tensor._prepare_init(process_group=process_group, init_rrefs=init_rrefs)

        # run post initialization, i.e. map registration, rpc initialization
        sharded_tensor._post_init()
        return sharded_tensor

    ShardedTensor._init_from_local_shards_and_global_metadata = _init_from_local_shards_and_global_metadata

except ImportError:
    pass

logger = logging.getLogger(__name__)

__all__ = ["save_checkpoint"]


def load_checkpoint(ddp_model, optimizer, opt_param_scheduler, checkpointing_context, skip_load_to_model_and_opt):
    # ref: how megatron `load_checkpoint` gets directory
    args = get_args()
    load_path = args.load

    assert Path(load_path).exists() and _is_dir_nonempty(
        load_path
    ), f"{args.load=} does not exist or is an empty directory. Did you specify the wrong folder?"

    if _is_megatron_checkpoint(load_path):
        return _load_checkpoint_megatron(
            ddp_model=ddp_model,
            optimizer=optimizer,
            opt_param_scheduler=opt_param_scheduler,
            checkpointing_context=checkpointing_context,
            skip_load_to_model_and_opt=skip_load_to_model_and_opt,
        )
    else:
        return _load_checkpoint_hf(
            ddp_model=ddp_model,
            optimizer=optimizer,
            args=args,
            load_path=load_path,
        )


def _is_megatron_checkpoint(path: str | Path) -> bool:
    return (Path(path) / "latest_checkpointed_iteration.txt").is_file() or bool(
        re.fullmatch(r"iter_\d{7}", Path(path).name)
    )


def _load_checkpoint_hf(ddp_model, optimizer, args, load_path: str):
    assert args.megatron_to_hf_mode == "bridge", "Only bridge mode is supported for loading HF checkpoint"
    from megatron.bridge import AutoBridge

    import slime_plugins.megatron_bridge  # noqa: F401

    logger.info(f"Load checkpoint from HuggingFace model into Megatron (path={load_path})")

    with megatron_bridge_utils.patch_megatron_model(ddp_model):
        bridge = AutoBridge.from_hf_pretrained(load_path, trust_remote_code=True)
        with _patch_bridge_skip_critic_output_layer(ddp_model, args):
            bridge.load_hf_weights(ddp_model)
    _maybe_load_critic_head_from_hf(ddp_model=ddp_model, args=args, load_path=load_path)

    # Copied from Megatron-core :: load_checkpoint (with simplifications)
    if (args.fp16 or args.bf16) and optimizer is not None:
        assert not args.load_main_params_from_ckpt
        optimizer.reload_model_params()

    # We can see `successfully loaded checkpoint from ... [ t 1/2, p 1/1 ] at iteration 0`
    # when loading Megatron, thus it is 0
    iteration = 0
    num_floating_point_operations_so_far = 0
    return iteration, num_floating_point_operations_so_far


def _unwrap_model_chunk(ddp_model):
    """Best-effort unwrap for [DDP] -> Megatron model chunk."""
    try:
        from megatron.core.utils import unwrap_model

        return unwrap_model(ddp_model)[0]
    except Exception:
        model_chunk = ddp_model[0] if isinstance(ddp_model, (list, tuple)) else ddp_model
        return getattr(model_chunk, "module", model_chunk)


def _get_model_role(ddp_model, model_chunk):
    role = getattr(model_chunk, "role", None)
    if role is None:
        wrapper = ddp_model[0] if isinstance(ddp_model, (list, tuple)) else ddp_model
        role = getattr(wrapper, "role", None)
    return role


@contextmanager
def _patch_bridge_skip_critic_output_layer(ddp_model, args):
    """
    For custom critic heads, output_layer is a value head rather than the
    language-model lm_head and should not be loaded from the base model.
    Mark embeddings as tied during Bridge HF load so Bridge skips output_layer.*.
    """
    model_chunk = _unwrap_model_chunk(ddp_model)
    if _get_model_role(ddp_model, model_chunk) != "critic":
        yield
        return

    if (
        getattr(args, "critic_value_loss_type", "mse") != "hl_gauss"
        and not getattr(args, "critic_separate_v0_head", False)
    ):
        yield
        return

    output_layer = getattr(model_chunk, "output_layer", None)
    model_config = getattr(model_chunk, "config", None)
    if not isinstance(output_layer, torch.nn.Linear) or model_config is None:
        yield
        return

    had_config_attr = hasattr(model_config, "share_embeddings_and_output_weights")
    old_config_value = getattr(model_config, "share_embeddings_and_output_weights", None)
    had_model_attr = hasattr(model_chunk, "share_embeddings_and_output_weights")
    old_model_value = getattr(model_chunk, "share_embeddings_and_output_weights", None)

    try:
        model_config.share_embeddings_and_output_weights = True
        if had_model_attr:
            model_chunk.share_embeddings_and_output_weights = True
        logger.info("Temporarily skip Bridge loading of custom critic output_layer; it will be kept initialized.")
        yield
    finally:
        if had_config_attr:
            model_config.share_embeddings_and_output_weights = old_config_value
        else:
            delattr(model_config, "share_embeddings_and_output_weights")
        if had_model_attr:
            model_chunk.share_embeddings_and_output_weights = old_model_value


def _load_lm_head_from_hf_safetensors(load_path: str | Path):
    """Load lm_head.{weight,bias} tensors from HF safetensors checkpoint."""
    path = Path(load_path)
    index_path = path / "model.safetensors.index.json"

    try:
        from safetensors import safe_open
    except Exception:
        logger.warning("safetensors is unavailable; skip loading critic lm_head from HF checkpoint.")
        return None, None

    weight = None
    bias = None

    # Sharded safetensors (recommended HF format).
    if index_path.is_file():
        with index_path.open("r", encoding="utf-8") as f:
            index = json.load(f)
        weight_map = index.get("weight_map", {})
        weight_file = weight_map.get("lm_head.weight")
        bias_file = weight_map.get("lm_head.bias")

        if weight_file is not None:
            with safe_open(path / weight_file, framework="pt", device="cpu") as sf:
                if "lm_head.weight" in sf.keys():
                    weight = sf.get_tensor("lm_head.weight")
        if bias_file is not None:
            with safe_open(path / bias_file, framework="pt", device="cpu") as sf:
                if "lm_head.bias" in sf.keys():
                    bias = sf.get_tensor("lm_head.bias")

        return weight, bias

    # Single safetensors file fallback.
    single_path = path / "model.safetensors"
    if single_path.is_file():
        with safe_open(single_path, framework="pt", device="cpu") as sf:
            if "lm_head.weight" in sf.keys():
                weight = sf.get_tensor("lm_head.weight")
            if "lm_head.bias" in sf.keys():
                bias = sf.get_tensor("lm_head.bias")
        return weight, bias

    return None, None


def _maybe_load_critic_head_from_hf(ddp_model, args, load_path: str):
    """
    Bridge may skip custom critic output head (e.g. output_layer for HL-Gauss).
    Explicitly copy HF lm_head.{weight,bias} -> Megatron output_layer when shapes match.
    """
    model_chunk = _unwrap_model_chunk(ddp_model)

    role = getattr(model_chunk, "role", None)
    if role is None:
        wrapper = ddp_model[0] if isinstance(ddp_model, (list, tuple)) else ddp_model
        role = getattr(wrapper, "role", None)
    if role != "critic":
        return

    output_layer = getattr(model_chunk, "output_layer", None)
    if not isinstance(output_layer, torch.nn.Linear):
        logger.warning("Critic model has no torch.nn.Linear output_layer; skip manual critic-head load.")
        return

    weight, bias = _load_lm_head_from_hf_safetensors(load_path)
    if weight is None:
        logger.warning("lm_head.weight not found in HF checkpoint; keep critic output_layer initialization as-is.")
        return

    if tuple(weight.shape) != tuple(output_layer.weight.shape):
        logger.warning(
            "Skip manual critic-head load due to shape mismatch: hf lm_head.weight=%s vs output_layer.weight=%s",
            tuple(weight.shape),
            tuple(output_layer.weight.shape),
        )
        return

    with torch.no_grad():
        output_layer.weight.copy_(weight.to(device=output_layer.weight.device, dtype=output_layer.weight.dtype))
        if output_layer.bias is not None and bias is not None:
            if tuple(bias.shape) == tuple(output_layer.bias.shape):
                output_layer.bias.copy_(bias.to(device=output_layer.bias.device, dtype=output_layer.bias.dtype))
            else:
                logger.warning(
                    "Skip copying lm_head.bias due to shape mismatch: hf=%s vs output_layer=%s",
                    tuple(bias.shape),
                    tuple(output_layer.bias.shape),
                )

    logger.info(
        "Manually loaded critic output_layer from HF lm_head: weight=%s bias=%s",
        tuple(output_layer.weight.shape),
        tuple(output_layer.bias.shape) if output_layer.bias is not None else None,
    )


def _is_dir_nonempty(path):
    with os.scandir(path) as it:
        return any(it)
