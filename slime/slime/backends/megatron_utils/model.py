import dataclasses
import gc
import logging
import math
import os
from argparse import Namespace
from collections.abc import Callable, Sequence
from functools import partial
from pathlib import Path

import torch
from megatron.core import mpu
from megatron.core.distributed import DistributedDataParallel as DDP
from megatron.core.distributed import finalize_model_grads
from megatron.core.enums import ModelType
from megatron.core.models.gpt import GPTModel
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.optimizer.optimizer import MegatronOptimizer
from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.core.utils import get_model_config
from megatron.training.global_vars import get_args
from megatron.training.training import get_model

try:
    from megatron.core.pipeline_parallel.utils import unwrap_model
except ImportError:
    from megatron.core.utils import unwrap_model
from slime.utils import logging_utils
from slime.utils.metric_utils import (
    EXPLAINED_VARIANCE_STAT_NAMES,
    VALUE_EXPLAINED_VARIANCE_STAT_PREFIX,
    compute_explained_variance_from_stats,
)
from slime.utils.memory_utils import clear_memory
from slime.utils.vapo import get_vapo_loss_normalizers

from .checkpoint import load_checkpoint, save_checkpoint
from .data import DataIterator, get_batch, get_data_iterator
from .loss import full_model_score_gradient_loss, loss_function
from .model_provider import get_model_provider_func, wrap_model_provider_with_freeze

logger = logging.getLogger(__name__)


def _finalize_value_explained_variance(loss_reduced: dict[str, float | torch.Tensor]) -> None:
    present = [
        f"{VALUE_EXPLAINED_VARIANCE_STAT_PREFIX}{name}" in loss_reduced
        for name in EXPLAINED_VARIANCE_STAT_NAMES
    ]
    if not any(present):
        return
    if not all(present):
        missing = [
            name
            for name, is_present in zip(EXPLAINED_VARIANCE_STAT_NAMES, present, strict=True)
            if not is_present
        ]
        raise KeyError(f"Missing value explained-variance statistics after reduction: {missing}")

    stats = {
        name: loss_reduced.pop(f"{VALUE_EXPLAINED_VARIANCE_STAT_PREFIX}{name}")
        for name in EXPLAINED_VARIANCE_STAT_NAMES
    }
    loss_reduced["vf_explained_var"] = compute_explained_variance_from_stats(stats)


def _iter_output_layers(model: Sequence[DDP]):
    for chunk_id, module in enumerate(unwrap_model(model)):
        output_layer = getattr(module, "output_layer", None)
        if output_layer is not None:
            yield chunk_id, output_layer


def _output_layer_parameters(model: Sequence[DDP]) -> list[torch.nn.Parameter]:
    """Return parameters belonging to the actor's untied output layer."""
    parameters: list[torch.nn.Parameter] = []
    for _chunk_id, output_layer in _iter_output_layers(model):
        parameters.extend(parameter for parameter in output_layer.parameters() if parameter.requires_grad)
    if not parameters:
        raise RuntimeError("analysis requires an untied actor output_layer with trainable parameters")
    return parameters


def _critic_output_layer_needs_reinit(args: Namespace, model: Sequence[DDP], role: str) -> bool:
    if role != "critic" or args.load is None:
        return False
    if getattr(args, "critic_output_layer_init_mode", "auto_reinit") == "legacy_v024":
        logger.warning(
            "Using legacy_v024 critic output layer init: keep constructor-initialized "
            "output_layer when checkpoint loading skips or mismatches it."
        )
        return False

    from megatron.core.dist_checkpointing.serialization import load_tensors_metadata
    from megatron.training.checkpointing import get_load_checkpoint_path_by_args

    checkpoint_path = Path(get_load_checkpoint_path_by_args(args))
    if not (checkpoint_path / ".metadata").is_file():
        return False

    checkpoint_metadata = load_tensors_metadata(str(checkpoint_path))
    for _chunk_id, output_layer in _iter_output_layers(model):
        for name in ("weight", "bias"):
            param = getattr(output_layer, name, None)
            if param is None:
                continue

            param_name = f"output_layer.{name}"
            ckpt_tensor_metadata = next(
                (
                    tensor_metadata
                    for key, tensor_metadata in checkpoint_metadata.items()
                    if key == param_name or key.endswith(f".{param_name}")
                ),
                None,
            )
            expected_shape = tuple(param.shape)
            checkpoint_shape = tuple(
                ckpt_tensor_metadata.global_shape) if ckpt_tensor_metadata is not None else None
            if checkpoint_shape == expected_shape:
                continue

            reason = (
                "missing from checkpoint metadata"
                if checkpoint_shape is None
                else f"shape mismatch checkpoint={checkpoint_shape} runtime={expected_shape}"
            )
            logger.warning(
                "Will reinitialize critic %s after checkpoint load because it is %s",
                param_name,
                reason,
            )
            return True

    return False


@torch.no_grad()
def _reinitialize_critic_output_layer(model: Sequence[DDP], *, zero_init: bool = False) -> None:
    for _chunk_id, output_layer in _iter_output_layers(model):
        if zero_init:
            output_layer.weight.data.zero_()
        else:
            output_layer.weight.data.normal_(mean=0.0, std=0.02)
        if output_layer.bias is not None:
            output_layer.bias.data.zero_()


def get_optimizer_param_scheduler(args: Namespace, optimizer: MegatronOptimizer) -> OptimizerParamScheduler:
    """Create and configure the optimizer learning-rate/weight-decay scheduler.

    This configures iteration-based schedules derived from the global batch size
    and run-time arguments.

    Args:
        args (Namespace): Training/runtime arguments (argparse namespace).
        optimizer (MegatronOptimizer): Megatron optimizer bound to the model.

    Returns:
        OptimizerParamScheduler: Initialized scheduler bound to ``optimizer``.
    """
    # Iteration-based training.
    args.train_iters = args.num_rollout * args.rollout_batch_size * \
        args.n_samples_per_prompt // args.global_batch_size
    if args.lr_decay_iters is None:
        args.lr_decay_iters = args.train_iters
    lr_decay_steps = args.lr_decay_iters * args.global_batch_size
    wd_incr_steps = args.train_iters * args.global_batch_size
    wsd_decay_steps = None
    if args.lr_wsd_decay_iters is not None:
        wsd_decay_steps = args.lr_wsd_decay_iters * args.global_batch_size
    if args.lr_warmup_fraction is not None:
        lr_warmup_steps = args.lr_warmup_fraction * lr_decay_steps
    else:
        lr_warmup_steps = args.lr_warmup_iters * args.global_batch_size

    opt_param_scheduler = OptimizerParamScheduler(
        optimizer,
        init_lr=args.lr_warmup_init,
        max_lr=args.lr,
        min_lr=args.min_lr,
        lr_warmup_steps=lr_warmup_steps,
        lr_decay_steps=lr_decay_steps,
        lr_decay_style=args.lr_decay_style,
        start_wd=args.start_weight_decay,
        end_wd=args.end_weight_decay,
        wd_incr_steps=wd_incr_steps,
        wd_incr_style=args.weight_decay_incr_style,
        use_checkpoint_opt_param_scheduler=args.use_checkpoint_opt_param_scheduler,
        override_opt_param_scheduler=args.override_opt_param_scheduler,
        wsd_decay_steps=wsd_decay_steps,
        lr_wsd_decay_style=args.lr_wsd_decay_style,
    )

    return opt_param_scheduler


def setup_model_and_optimizer(
    args: Namespace,
    role: str = "actor",
) -> tuple[list[DDP], MegatronOptimizer, OptimizerParamScheduler]:
    """Build model(s), wrap with DDP, and construct optimizer and scheduler.

    Args:
        args (Namespace): Training/runtime arguments (argparse namespace).
        role (str): Logical role of the model (e.g., "actor", "critic").
        no_wd_decay_cond (Callable[..., bool] | None): Predicate to exclude
            parameters from weight decay.
        scale_lr_cond (Callable[..., bool] | None): Predicate to scale LR for
            selected parameter groups.
        lr_mult (float): Global learning-rate multiplier for the optimizer.

    Returns:
        tuple[list[DDP], MegatronOptimizer, OptimizerParamScheduler]:
            - List of model chunks wrapped by ``DDP``.
            - The constructed ``MegatronOptimizer`` instance.
            - The learning-rate/weight-decay scheduler tied to the optimizer.
    """
    assert not args.moe_use_upcycling
    assert args.load is not None or args.pretrained_checkpoint is not None

    model = get_model(
        wrap_model_provider_with_freeze(get_model_provider_func(
            args, role), args), ModelType.encoder_or_decoder
    )

    # Optimizer
    kwargs = {}
    for f in dataclasses.fields(OptimizerConfig):
        if hasattr(args, f.name):
            kwargs[f.name] = getattr(args, f.name)
    config = OptimizerConfig(**kwargs)
    config.timers = None

    optimizer = get_megatron_optimizer(
        config=config,
        model_chunks=model,
        use_gloo_process_groups=args.enable_gloo_process_groups,
    )
    opt_param_scheduler = get_optimizer_param_scheduler(args, optimizer)
    return model, optimizer, opt_param_scheduler


def enable_forward_pre_hook(model_chunks: Sequence[DDP]) -> None:
    """Enable forward pre-hooks for provided DDP-wrapped model chunks.

    Args:
        model_chunks (Sequence[DDP]): Sequence of DDP modules to enable hooks on.
    """
    for model_chunk in model_chunks:
        assert isinstance(model_chunk, DDP)
        model_chunk.enable_forward_pre_hook()


def disable_forward_pre_hook(model_chunks: Sequence[DDP], param_sync: bool = True) -> None:
    """Disable forward pre-hooks for provided DDP-wrapped model chunks.

    Args:
        model_chunks (Sequence[DDP]): Sequence of DDP modules to disable hooks on.
        param_sync (bool): Whether to synchronize parameters when disabling.
    """
    for model_chunk in model_chunks:
        assert isinstance(model_chunk, DDP)
        model_chunk.disable_forward_pre_hook(param_sync=param_sync)


@torch.no_grad()
def forward_only(
    f: Callable[..., dict[str, list[torch.Tensor]]],
    args: Namespace,
    model: Sequence[DDP],
    data_iterator: Sequence[DataIterator],
    num_microbatches: Sequence[int],
    store_prefix: str = "",
) -> dict[str, list[torch.Tensor]]:
    """Run forward passes only and collect non-loss outputs (e.g., logprobs).

    The model is put into evaluation mode, a forward-only pipeline pass is
    executed, and relevant outputs are aggregated and returned.

    Args:
        f (Callable[..., dict[str, list[torch.Tensor]]]): Post-forward callback used to
            compute and package outputs to collect. This should accept a logits
            tensor as its first positional argument and additional keyword-only
            arguments; see ``get_log_probs_and_entropy``/``get_values`` in
            ``megatron_utils.loss`` for examples. It will be partially applied
            so that the callable returned from the internal forward step only
            requires the logits tensor.
        args (Namespace): Runtime arguments.
        model (Sequence[DDP]): Sequence of DDP-wrapped model chunks.
        data_iterator (Sequence[DataIterator]): Iterable(s) yielding batches for inference.
        num_microbatches (Sequence[int]): Number of microbatches per rollout step.
        store_prefix (str): Prefix to prepend to stored output keys.

    Returns:
        dict[str, list[torch.Tensor]]: Aggregated outputs keyed by ``store_prefix + key``.
    """

    # reset data iterator
    for iterator in data_iterator:
        iterator.reset()

    config = get_model_config(model[0])

    def forward_step(
        data_iterator: DataIterator, model: GPTModel, return_schedule_plan: bool = False
    ) -> tuple[torch.Tensor, Callable[[torch.Tensor], dict[str, list[torch.Tensor]]]]:
        """Forward step used by Megatron's pipeline engine.

        Args:
            data_iterator (DataIterator): Input data iterator.
            model (GPTModel): The GPT model chunk to execute.

        Returns:
            tuple[torch.Tensor, Callable[[torch.Tensor], dict[str, list[torch.Tensor]]]]:
            Output tensor(s) and a callable that computes and packages results
            to be collected by the engine.
        """

        assert not return_schedule_plan, "forward_only step should never return schedule plan"

        # Get the batch.
        batch = get_batch(
            data_iterator,
            [
                "tokens",
                "loss_masks",
                "multimodal_train_inputs",
                "total_lengths",
                "response_lengths",
                "sample_indices",
                "max_seq_lens",
                "hlg_debug_distribution_mask",
                "mc_value_label_masks",
            ],
            args.data_pad_size_multiplier,
            args.qkv_format,
            args.allgather_cp,
        )
        unconcat_tokens = batch["unconcat_tokens"]
        tokens = batch["tokens"]
        packed_seq_params = batch["packed_seq_params"]
        total_lengths = batch["total_lengths"]
        response_lengths = batch["response_lengths"]
        output_tensor = model(
            input_ids=tokens,
            position_ids=None,
            attention_mask=None,
            labels=None,
            packed_seq_params=packed_seq_params,
            loss_mask=batch["full_loss_masks"],
            **(batch["multimodal_train_inputs"] if batch["multimodal_train_inputs"] is not None else {}),
        )

        analysis_callback_kwargs = {}
        if bool(getattr(args, "_analysis_capture_anchor_hidden", False)):
            analysis_callback_kwargs["mc_value_label_masks"] = batch.get("mc_value_label_masks", None)

        return output_tensor, partial(
            f,
            args=args,
            unconcat_tokens=unconcat_tokens,
            total_lengths=total_lengths,
            response_lengths=response_lengths,
            sample_indices=batch.get("sample_indices", None),
            with_entropy=args.use_rollout_entropy,
            max_seq_lens=batch.get("max_seq_lens", None),
            hlg_debug_distribution_mask=batch.get(
                "hlg_debug_distribution_mask", None),
            **analysis_callback_kwargs,
        )

    # Turn on evaluation mode which disables dropout.
    for model_module in model:
        model_module.eval()

    if args.custom_megatron_before_log_prob_hook_path:
        from slime.utils.misc import load_function

        custom_before_log_prob_hook = load_function(
            args.custom_megatron_before_log_prob_hook_path)
        custom_before_log_prob_hook(args, model, store_prefix)

    forward_backward_func = get_forward_backward_func()
    # Don't care about timing during evaluation
    config.timers = None
    forward_data_store = []
    num_steps_per_rollout = len(num_microbatches)
    for step_id in range(num_steps_per_rollout):
        forward_data_store += forward_backward_func(
            forward_step_func=forward_step,
            data_iterator=data_iterator,
            model=model,
            num_microbatches=num_microbatches[step_id],
            seq_length=args.seq_length,
            micro_batch_size=args.micro_batch_size,
            forward_only=True,
        )

    if bool(getattr(args, "_analysis_capture_anchor_hidden", False)):
        unconsumed_hidden = getattr(args, "_analysis_anchor_hidden_buffer", [])
        if unconsumed_hidden:
            raise RuntimeError(
                "anchor hidden-state capture did not match forward callbacks: "
                f"{len(unconsumed_hidden)} hidden microbatches remain"
            )

    # Move model back to the train mode.
    for model_module in model:
        model_module.train()

    rollout_data = {}
    # Store the results on the last stage
    if mpu.is_pipeline_last_stage():
        keys = forward_data_store[0].keys()
        for key in keys:
            values = []
            for value in forward_data_store:
                assert isinstance(value[key], list)
                values += value[key]

            if args.use_dynamic_batch_size:
                # TODO: This is ugly... Find a better way to make the data have the same order.
                # TODO: move this out of the loop.
                origin_values = [None] * len(values)
                origin_indices = sum(data_iterator[0].micro_batch_indices, [])
                for value, origin_index in zip(values, origin_indices, strict=False):
                    origin_values[origin_index] = value
                values = origin_values
            rollout_data[f"{store_prefix}{key}"] = values
    return rollout_data


def _add_anchor_advantage_gradient_data(args: Namespace, rollout_data: dict) -> dict:
    """Build critic/oracle advantage coefficients on identical MC anchors."""
    if float(getattr(args, "gamma", 1.0)) != 1.0 or float(getattr(args, "lambd", 1.0)) != 1.0:
        raise ValueError("anchor advantage gradient analysis requires gamma=1 and lambda=1")
    if float(getattr(args, "kl_coef", 0.0)) != 0.0:
        raise ValueError("anchor advantage gradient analysis requires kl_coef=0")
    required = ["returns", "values", "mc_value_labels", "mc_value_label_masks", "loss_masks"]
    missing = [key for key in required if rollout_data.get(key) is None]
    if missing:
        raise KeyError(f"anchor advantage gradient analysis requires rollout fields: {missing}")

    analysis_data = dict(rollout_data)
    critic_advantages = []
    oracle_advantages = []
    advantage_errors = []
    anchor_masks = []
    for returns, values, mc_values, mc_mask, loss_mask in zip(
        rollout_data["returns"],
        rollout_data["values"],
        rollout_data["mc_value_labels"],
        rollout_data["mc_value_label_masks"],
        rollout_data["loss_masks"],
        strict=True,
    ):
        returns_tensor = torch.as_tensor(returns)
        values_tensor = torch.as_tensor(values, device=returns_tensor.device, dtype=returns_tensor.dtype)
        mc_tensor = torch.as_tensor(mc_values, device=returns_tensor.device, dtype=returns_tensor.dtype)
        mc_mask_tensor = torch.as_tensor(mc_mask, device=returns_tensor.device, dtype=returns_tensor.dtype)
        loss_mask_tensor = torch.as_tensor(loss_mask, device=returns_tensor.device, dtype=returns_tensor.dtype)
        expected = returns_tensor.numel()
        lengths = [
            values_tensor.numel(),
            mc_tensor.numel(),
            mc_mask_tensor.numel(),
            loss_mask_tensor.numel(),
        ]
        if any(length != expected for length in lengths):
            raise ValueError(
                "anchor advantage gradient length mismatch: "
                f"returns={expected}, values/mc/mc_mask/loss_mask={lengths}"
            )
        critic = returns_tensor.reshape(-1) - values_tensor.reshape(-1)
        oracle = returns_tensor.reshape(-1) - mc_tensor.reshape(-1)
        critic_advantages.append(critic)
        oracle_advantages.append(oracle)
        advantage_errors.append(critic - oracle)
        anchor_masks.append(loss_mask_tensor.reshape(-1) * (mc_mask_tensor.reshape(-1) != 0).to(loss_mask_tensor.dtype))

    if not any(float(mask.sum().item()) > 0.0 for mask in anchor_masks):
        raise ValueError("anchor advantage gradient analysis found no valid MC anchors")
    analysis_data["analysis_critic_anchor_advantages"] = critic_advantages
    analysis_data["analysis_oracle_anchor_advantages"] = oracle_advantages
    analysis_data["analysis_anchor_advantage_errors"] = advantage_errors
    analysis_data["analysis_anchor_masks"] = anchor_masks
    return analysis_data


def _interpolate_mc_values(mc_values: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
    """Linear MC interpolation at every response token; extend end anchors constantly."""
    positions = torch.nonzero(anchors, as_tuple=False).reshape(-1)
    if positions.numel() == 0:
        raise ValueError("MC interpolation requires at least one valid anchor per response")
    target = torch.arange(mc_values.numel(), device=mc_values.device)
    right = torch.searchsorted(positions, target).clamp(max=positions.numel() - 1)
    left = (right - 1).clamp(min=0)
    left_pos = positions[left]
    right_pos = positions[right]
    fraction = (target - left_pos).to(mc_values.dtype) / (right_pos - left_pos).clamp_min(1).to(mc_values.dtype)
    fraction = fraction.clamp(0.0, 1.0)
    return mc_values[left] + fraction * (mc_values[right] - mc_values[left])


def _add_clipped_gradient_data(args: Namespace, rollout_data: dict, selected_samples: int) -> tuple[dict, dict]:
    """Freeze the local derivative of the PPO surrogate at the replay checkpoint."""
    current = rollout_data.get("_analysis_current_log_probs")
    if current is None:
        raise KeyError("clipped gradient analysis requires current actor token log-probs")
    old = rollout_data.get("rollout_log_probs") if getattr(args, "use_rollout_logprobs", False) else rollout_data.get("_analysis_old_log_probs")
    ratio_source = "rollout_log_probs" if getattr(args, "use_rollout_logprobs", False) else rollout_data.get(
        "_analysis_old_log_probs_source", "replayed_old_log_probs"
    )
    if old is None:
        if getattr(args, "use_rollout_logprobs", False):
            raise KeyError("--use-rollout-logprobs requires real rollout_log_probs for clipping")
        old = current
        ratio_source = "current_checkpoint_as_old_ratio_one"
    use_tis = bool(getattr(args, "analysis_clipped_gradient_use_tis", False))
    rollout_probs = rollout_data.get("rollout_log_probs")
    if use_tis and (rollout_probs is None or rollout_data.get("_analysis_old_log_probs") is None):
        raise KeyError("clipped gradient TIS requires genuine rollout_log_probs and old-policy log_probs")
    if getattr(args, "custom_tis_function_path", None) and use_tis:
        raise RuntimeError("clipped gradient analysis supports Slime's default vanilla TIS only")
    if getattr(args, "use_opsm", False) or getattr(args, "advantage_estimator", "") == "gspo":
        raise RuntimeError("clipped gradient analysis currently supports token-level PPO without OPSM")
    if getattr(args, "calculate_per_token_loss", False):
        raise RuntimeError("clipped gradient analysis requires Slime's per-response token-mean loss reducer")
    if getattr(args, "custom_pg_loss_reducer_function_path", None):
        raise RuntimeError("clipped gradient analysis requires the default policy-loss reducer")
    eps_low = float(args.eps_clip)
    eps_high = float(args.eps_clip_high)
    dual = getattr(args, "eps_clip_c", None)
    dual = None if dual is None else float(dual)
    if dual is not None and dual <= 1.0:
        raise ValueError("eps_clip_c must exceed 1")
    data = dict(rollout_data)
    keys = (
        "analysis_clipped_critic_anchor", "analysis_clipped_oracle_anchor", "analysis_clipped_anchor_error",
        "analysis_clipped_critic_full_token", "analysis_clipped_oracle_full_token", "analysis_clipped_full_token_error",
    )
    for key in keys:
        data[key] = []
    counts = {"anchor": 0, "full_token": 0, "critic_anchor": 0, "oracle_anchor": 0,
              "critic_full_token": 0, "oracle_full_token": 0, "mc_anchors": 0, "tis_truncated": 0}
    ratio_sum = 0.0
    tis_sum = 0.0
    sample_weights = rollout_data.get("sample_weights")
    for index in range(len(rollout_data["response_lengths"])):
        returns = torch.as_tensor(rollout_data["returns"][index]).detach().float().reshape(-1)
        device = returns.device
        def tensor(key: str) -> torch.Tensor:
            return torch.as_tensor(rollout_data[key][index], device=device).detach().float().reshape(-1)
        values, mc, mc_mask, loss_mask = (tensor(key) for key in ("values", "mc_value_labels", "mc_value_label_masks", "loss_masks"))
        cur = torch.as_tensor(current[index], device=device).detach().float().reshape(-1)
        prior = torch.as_tensor(old[index], device=device).detach().float().reshape(-1)
        if any(item.numel() != returns.numel() for item in (values, mc, mc_mask, loss_mask, cur, prior)):
            raise ValueError(f"clipped gradient response length mismatch at sample {index}")
        sample_weight = 1.0 if sample_weights is None else float(sample_weights[index])
        if not math.isfinite(sample_weight):
            raise ValueError(f"non-finite sample weight at sample {index}")
        valid = loss_mask != 0
        anchors = valid & (mc_mask != 0)
        interpolated = _interpolate_mc_values(mc, anchors)
        ratio = torch.exp(cur - prior)
        if not torch.isfinite(ratio).all():
            raise FloatingPointError(f"non-finite PPO ratio at sample {index}")
        weight = torch.ones_like(ratio)
        if use_tis:
            rollout = torch.as_tensor(rollout_probs[index], device=device).detach().float().reshape(-1)
            train_old = torch.as_tensor(rollout_data["_analysis_old_log_probs"][index], device=device).detach().float().reshape(-1)
            if rollout.numel() != returns.numel() or train_old.numel() != returns.numel():
                raise ValueError(f"TIS token log-prob length mismatch at sample {index}")
            tis = torch.exp(train_old - rollout)
            if not torch.isfinite(tis).all():
                raise FloatingPointError(f"non-finite TIS ratio at sample {index}")
            weight = tis.clamp(min=float(args.tis_clip_low), max=float(args.tis_clip))
            if index < selected_samples:
                counts["tis_truncated"] += int(((tis != weight) & valid).sum().item())
                tis_sum += float((weight * valid).sum().item())

        def post(advantage: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            unclipped_loss = -ratio * advantage
            clipped_loss = -ratio.clamp(1.0 - eps_low, 1.0 + eps_high) * advantage
            standard_clip = clipped_loss > unclipped_loss
            chosen = torch.where(standard_clip, clipped_loss, unclipped_loss)
            dual_clip = torch.zeros_like(standard_clip)
            if dual is not None:
                dual_clip = (advantage < 0) & ((-dual * advantage) < chosen)
            inactive = standard_clip | dual_clip
            return torch.where(inactive, 0.0, ratio * advantage * weight * sample_weight), inactive

        critic, critic_inactive = post(returns - values)
        oracle_anchor, anchor_inactive = post(returns - mc)
        oracle_full, full_inactive = post(returns - interpolated)
        for key, coefficient in zip(keys, (
            critic, oracle_anchor, critic - oracle_anchor,
            critic, oracle_full, critic - oracle_full,
        ), strict=True):
            data[key].append(coefficient)
        if index < selected_samples:
            counts["anchor"] += int(anchors.sum().item())
            counts["full_token"] += int(valid.sum().item())
            counts["mc_anchors"] += int(anchors.sum().item())
            counts["critic_anchor"] += int((critic_inactive & anchors).sum().item())
            counts["oracle_anchor"] += int((anchor_inactive & anchors).sum().item())
            counts["critic_full_token"] += int((critic_inactive & valid).sum().item())
            counts["oracle_full_token"] += int((full_inactive & valid).sum().item())
            ratio_sum += float((ratio * valid).sum().item())
    metadata = {"ppo_ratio_source": ratio_source, "clipped_gradient_tis_enabled": use_tis,
                "mc_interpolation": "linear_between_valid_anchors_constant_at_edges",
                "eps_clip": eps_low, "eps_clip_high": eps_high, "eps_clip_c": dual,
                "clipped_gradient_normalization": "weighted_mean_of_response_token_means",
                "ppo_ratio_sum_local": ratio_sum, "tis_weight_sum_local": tis_sum}
    metadata.update({f"clipped_count_{key}_local": value for key, value in counts.items()})
    return data, metadata


def compute_full_model_gradient_statistics(
    args: Namespace,
    model: Sequence[DDP],
    optimizer: MegatronOptimizer,
    rollout_data: dict,
) -> dict[str, object]:
    """Run exact full-model score-gradient backwards without updating weights.

    Three objectives are differentiated: return, baseline, and advantage. For
    each one we measure the norm of every fixed-size response group and the norm
    of the mean gradient over all selected responses. These scalar sufficient
    statistics are enough to recover leakage, cosine, variance, and SNR exactly
    without materializing an 8B-parameter gradient vector on disk.
    """
    if mpu.get_context_parallel_world_size() != 1:
        raise RuntimeError("exact full-model gradient analysis requires context_parallel_size == 1")
    if mpu.get_pipeline_model_parallel_world_size() != 1:
        raise RuntimeError("exact full-model gradient analysis currently requires pipeline_model_parallel_size == 1")

    total_local_samples = len(rollout_data["response_lengths"])
    if "returns" not in rollout_data or "advantages" not in rollout_data:
        raise KeyError("full-model gradient analysis requires returns and advantages")
    analysis_data = _add_anchor_advantage_gradient_data(args, rollout_data)
    analysis_data["analysis_baselines"] = [
        torch.as_tensor(ret) - torch.as_tensor(adv)
        for ret, adv in zip(rollout_data["returns"], rollout_data["advantages"], strict=True)
    ]
    requested_samples = int(getattr(args, "analysis_full_gradient_samples", 512))
    group_size = max(int(getattr(args, "analysis_full_gradient_microbatch_size", 256)), 1)
    selected_samples = total_local_samples if requested_samples <= 0 else min(total_local_samples, requested_samples)
    if selected_samples >= group_size:
        selected_samples = (selected_samples // group_size) * group_size
    else:
        group_size = selected_samples
    if selected_samples <= 0:
        raise ValueError("no responses available for exact full-model gradient analysis")

    # Every DP rank must execute the same number of backward schedules.
    selected_tensor = torch.tensor(selected_samples, dtype=torch.int64, device=torch.cuda.current_device())
    torch.distributed.all_reduce(
        selected_tensor,
        op=torch.distributed.ReduceOp.MIN,
        group=mpu.get_data_parallel_group(with_context_parallel=True),
    )
    selected_samples = int(selected_tensor.item())
    if selected_samples >= group_size:
        selected_samples = (selected_samples // group_size) * group_size
    else:
        group_size = selected_samples

    def subset(source: dict, start: int, end: int) -> dict:
        result = {}
        for key, value in source.items():
            if isinstance(value, (list, tuple)) and len(value) == total_local_samples:
                result[key] = list(value[start:end])
            else:
                result[key] = value
        result["dynamic_global_batch_size"] = (end - start) * mpu.get_data_parallel_world_size(
            with_context_parallel=True
        )
        return result

    def backward_norm(
        data: dict, coefficient_key: str, normalizer: int, mask_key: str = "loss_masks"
    ) -> float:
        data_iterators, microbatches_by_step = get_data_iterator(args, model, data)
        if len(microbatches_by_step) != 1:
            raise RuntimeError(
                "full-model gradient analysis expects exactly one statistical batch, "
                f"got {len(microbatches_by_step)}"
            )
        num_microbatches = int(microbatches_by_step[0])
        for iterator in data_iterators:
            iterator.reset()

        for model_chunk in model:
            model_chunk.zero_grad_buffer()
        optimizer.zero_grad()

        def forward_step(
            data_iterator: DataIterator,
            model_chunk: GPTModel,
            return_schedule_plan: bool = False,
        ):
            batch_keys = [
                "tokens",
                "multimodal_train_inputs",
                "total_lengths",
                "response_lengths",
                "loss_masks",
                coefficient_key,
                "max_seq_lens",
            ]
            if mask_key != "loss_masks":
                batch_keys.append(mask_key)
            batch = get_batch(
                data_iterator,
                batch_keys,
                args.data_pad_size_multiplier,
                args.qkv_format,
                args.allgather_cp,
            )
            forward_kwargs = {
                "input_ids": batch["tokens"],
                "position_ids": None,
                "attention_mask": None,
                "labels": None,
                "packed_seq_params": batch["packed_seq_params"],
                "loss_mask": batch["full_loss_masks"],
            }
            if batch["multimodal_train_inputs"] is not None:
                forward_kwargs.update(batch["multimodal_train_inputs"])
            if return_schedule_plan:
                if args.enable_mtp_training:
                    raise RuntimeError("full-model gradient analysis does not support MTP schedule plans")
                output_tensor = model_chunk.build_schedule_plan(**forward_kwargs)
            else:
                output_tensor = model_chunk(**forward_kwargs)
            return output_tensor, partial(
                full_model_score_gradient_loss,
                args,
                batch,
                coefficient_key,
                normalizer,
                mask_key,
            )

        get_forward_backward_func()(
            forward_step_func=forward_step,
            data_iterator=data_iterators,
            model=model,
            num_microbatches=num_microbatches,
            seq_length=args.seq_length,
            micro_batch_size=args.micro_batch_size,
            decoder_seq_length=args.decoder_seq_length,
            forward_only=False,
        )
        found_inf = optimizer.prepare_grads()
        if bool(found_inf):
            raise FloatingPointError(f"non-finite {coefficient_key} gradient in full-model analysis")
        norm = optimizer.get_grad_norm()
        norm_value = float(norm.item() if isinstance(norm, torch.Tensor) else norm)
        for model_chunk in model:
            model_chunk.zero_grad_buffer()
        optimizer.zero_grad()
        return norm_value

    coefficient_keys = {
        "return": ("returns", "loss_masks"),
        "baseline": ("analysis_baselines", "loss_masks"),
        "advantage": ("advantages", "loss_masks"),
        "critic_anchor": ("analysis_critic_anchor_advantages", "analysis_anchor_masks"),
        "oracle_anchor": ("analysis_oracle_anchor_advantages", "analysis_anchor_masks"),
        "anchor_error": ("analysis_anchor_advantage_errors", "analysis_anchor_masks"),
    }
    clipped_metadata = None
    if getattr(args, "analysis_clipped_gradient", False):
        analysis_data, clipped_metadata = _add_clipped_gradient_data(args, analysis_data, selected_samples)
        coefficient_keys.update({
            "clipped_critic_anchor": ("analysis_clipped_critic_anchor", "analysis_anchor_masks"),
            "clipped_oracle_anchor": ("analysis_clipped_oracle_anchor", "analysis_anchor_masks"),
            "clipped_anchor_error": ("analysis_clipped_anchor_error", "analysis_anchor_masks"),
            "clipped_critic_full_token": ("analysis_clipped_critic_full_token", "loss_masks"),
            "clipped_oracle_full_token": ("analysis_clipped_oracle_full_token", "loss_masks"),
            "clipped_full_token_error": ("analysis_clipped_full_token_error", "loss_masks"),
        })
    # Build subsets only after every requested coefficient has been added.
    # Otherwise clipped fields are absent from the pre-existing selected view.
    selected = subset(analysis_data, 0, selected_samples)
    group_ranges = [
        (start, start + group_size) for start in range(0, selected_samples, group_size)
    ]
    result: dict[str, object] = {
        "gradient_type": "exact_full_model",
        "num_samples_local": selected_samples,
        "num_samples_global": selected_samples
        * mpu.get_data_parallel_world_size(with_context_parallel=True),
        "microbatch_size_local": group_size,
        "num_gradient_batches": len(group_ranges),
    }
    if clipped_metadata is not None:
        result.update(clipped_metadata)

    config = get_model_config(model[0])
    original_finalize = config.finalize_model_grads_func
    original_grad_scale = config.grad_scale_func
    original_modes = [model_chunk.training for model_chunk in model]
    config.finalize_model_grads_func = finalize_model_grads
    config.grad_scale_func = optimizer.scale_loss
    for model_chunk in model:
        model_chunk.eval()
    try:
        for label, (key, mask_key) in coefficient_keys.items():
            group_norm_squares = []
            for start, end in group_ranges:
                norm = backward_norm(
                    subset(analysis_data, start, end), key, group_size, mask_key
                )
                group_norm_squares.append(norm * norm)
            mean_norm = backward_norm(selected, key, selected_samples, mask_key)
            result[f"mean_gradient_norm_{label}"] = mean_norm
            result[f"microbatch_gradient_norm_sq_{label}"] = group_norm_squares
    finally:
        config.finalize_model_grads_func = original_finalize
        config.grad_scale_func = original_grad_scale
        for model_chunk, was_training in zip(model, original_modes, strict=True):
            model_chunk.train(was_training)
        for model_chunk in model:
            model_chunk.zero_grad_buffer()
        optimizer.zero_grad()

    return result


def compute_output_layer_gradient_statistics(
    args: Namespace,
    model: Sequence[DDP],
    optimizer: MegatronOptimizer,
    rollout_data: dict,
) -> dict[str, object]:
    """Compute the output-layer component of a complete actor backward.

    The full model participates in the backward exactly as it does for the
    policy-gradient objective. After each backward, only the gradients attached
    to the untied ``output_layer`` (and its bias, when present) are measured.
    No hidden-state or selected-logit approximation is used.
    """
    if mpu.get_context_parallel_world_size() != 1:
        raise RuntimeError("exact output-layer gradient analysis requires context_parallel_size == 1")
    if mpu.get_pipeline_model_parallel_world_size() != 1:
        raise RuntimeError("exact output-layer gradient analysis requires pipeline_model_parallel_size == 1")
    if not bool(getattr(args, "untie_embeddings_and_output_weights", False)):
        raise RuntimeError(
            "exact output-layer gradient analysis requires --untie-embeddings-and-output-weights"
        )

    output_parameters = _output_layer_parameters(model)
    return _compute_gradient_statistics(
        args,
        model,
        optimizer,
        rollout_data,
        gradient_type="exact_output_layer",
        gradient_norm_fn=lambda: _output_layer_gradient_norm(output_parameters),
    )


def _output_layer_gradient_norm(parameters: list[torch.nn.Parameter]) -> float:
    local_squared_norm = torch.zeros(1, device=parameters[0].device, dtype=torch.float32)
    for parameter in parameters:
        gradient = getattr(parameter, "main_grad", None)
        if gradient is None:
            gradient = parameter.grad
        if gradient is not None:
            local_squared_norm += gradient.detach().float().pow(2).sum()
    # Output-layer weights are vocab-sharded across tensor parallel ranks and
    # replicated across data parallel ranks; reduce only over the TP group.
    torch.distributed.all_reduce(
        local_squared_norm,
        op=torch.distributed.ReduceOp.SUM,
        group=mpu.get_tensor_model_parallel_group(),
    )
    return float(local_squared_norm.sqrt().item())


def _compute_gradient_statistics(
    args: Namespace,
    model: Sequence[DDP],
    optimizer: MegatronOptimizer,
    rollout_data: dict,
    *,
    gradient_type: str,
    gradient_norm_fn: Callable[[], float],
) -> dict[str, object]:
    """Shared sufficient-statistics implementation for exact gradient scopes."""
    total_local_samples = len(rollout_data["response_lengths"])
    if "returns" not in rollout_data or "advantages" not in rollout_data:
        raise KeyError(f"{gradient_type} analysis requires returns and advantages")
    analysis_data = _add_anchor_advantage_gradient_data(args, rollout_data)
    analysis_data["analysis_baselines"] = [
        torch.as_tensor(ret) - torch.as_tensor(adv)
        for ret, adv in zip(rollout_data["returns"], rollout_data["advantages"], strict=True)
    ]
    requested_samples = int(getattr(args, "analysis_full_gradient_samples", 512))
    group_size = max(int(getattr(args, "analysis_full_gradient_microbatch_size", 256)), 1)
    selected_samples = total_local_samples if requested_samples <= 0 else min(total_local_samples, requested_samples)
    if selected_samples >= group_size:
        selected_samples = (selected_samples // group_size) * group_size
    else:
        group_size = selected_samples
    if selected_samples <= 0:
        raise ValueError("no responses available for exact gradient analysis")

    selected_tensor = torch.tensor(selected_samples, dtype=torch.int64, device=torch.cuda.current_device())
    torch.distributed.all_reduce(
        selected_tensor,
        op=torch.distributed.ReduceOp.MIN,
        group=mpu.get_data_parallel_group(with_context_parallel=True),
    )
    selected_samples = int(selected_tensor.item())
    if selected_samples >= group_size:
        selected_samples = (selected_samples // group_size) * group_size
    else:
        group_size = selected_samples

    def subset(source: dict, start: int, end: int) -> dict:
        result = {
            key: list(value[start:end])
            if isinstance(value, (list, tuple)) and len(value) == total_local_samples
            else value
            for key, value in source.items()
        }
        result["dynamic_global_batch_size"] = (end - start) * mpu.get_data_parallel_world_size(
            with_context_parallel=True
        )
        return result

    def backward_norm(
        data: dict, coefficient_key: str, normalizer: int, mask_key: str = "loss_masks"
    ) -> float:
        data_iterators, microbatches_by_step = get_data_iterator(args, model, data)
        if len(microbatches_by_step) != 1:
            raise RuntimeError(
                f"{gradient_type} analysis expects exactly one statistical batch, "
                f"got {len(microbatches_by_step)}"
            )
        num_microbatches = int(microbatches_by_step[0])
        for iterator in data_iterators:
            iterator.reset()
        for model_chunk in model:
            model_chunk.zero_grad_buffer()
        optimizer.zero_grad()

        def forward_step(
            data_iterator: DataIterator,
            model_chunk: GPTModel,
            return_schedule_plan: bool = False,
        ):
            batch_keys = [
                "tokens",
                "multimodal_train_inputs",
                "packed_seq_params",
                "total_lengths",
                "response_lengths",
                "loss_masks",
                coefficient_key,
                "max_seq_lens",
            ]
            if mask_key != "loss_masks":
                batch_keys.append(mask_key)
            batch = get_batch(
                data_iterator,
                batch_keys,
                args.data_pad_size_multiplier,
                args.qkv_format,
                args.allgather_cp,
            )
            forward_kwargs = {
                "input_ids": batch["tokens"],
                "position_ids": None,
                "attention_mask": None,
                "labels": None,
                "packed_seq_params": batch["packed_seq_params"],
                "loss_mask": batch["full_loss_masks"],
            }
            if batch["multimodal_train_inputs"] is not None:
                forward_kwargs.update(batch["multimodal_train_inputs"])
            output_tensor = (
                model_chunk.build_schedule_plan(**forward_kwargs)
                if return_schedule_plan
                else model_chunk(**forward_kwargs)
            )
            return output_tensor, partial(
                full_model_score_gradient_loss,
                args,
                batch,
                coefficient_key,
                normalizer,
                mask_key,
            )

        get_forward_backward_func()(
            forward_step_func=forward_step,
            data_iterator=data_iterators,
            model=model,
            num_microbatches=num_microbatches,
            seq_length=args.seq_length,
            micro_batch_size=args.micro_batch_size,
            decoder_seq_length=args.decoder_seq_length,
            forward_only=False,
        )
        found_inf = optimizer.prepare_grads()
        if bool(found_inf):
            raise FloatingPointError(f"non-finite {coefficient_key} gradient in {gradient_type} analysis")
        norm_value = gradient_norm_fn()
        for model_chunk in model:
            model_chunk.zero_grad_buffer()
        optimizer.zero_grad()
        return norm_value

    selected = subset(analysis_data, 0, selected_samples)
    group_ranges = [(start, start + group_size) for start in range(0, selected_samples, group_size)]
    coefficient_keys = {
        "return": ("returns", "loss_masks"),
        "baseline": ("analysis_baselines", "loss_masks"),
        "advantage": ("advantages", "loss_masks"),
        "critic_anchor": ("analysis_critic_anchor_advantages", "analysis_anchor_masks"),
        "oracle_anchor": ("analysis_oracle_anchor_advantages", "analysis_anchor_masks"),
        "anchor_error": ("analysis_anchor_advantage_errors", "analysis_anchor_masks"),
    }
    result: dict[str, object] = {
        "gradient_type": gradient_type,
        "num_samples_local": selected_samples,
        "num_samples_global": selected_samples * mpu.get_data_parallel_world_size(with_context_parallel=True),
        "microbatch_size_local": group_size,
        "num_gradient_batches": len(group_ranges),
    }
    config = get_model_config(model[0])
    original_finalize = config.finalize_model_grads_func
    original_grad_scale = config.grad_scale_func
    original_modes = [model_chunk.training for model_chunk in model]
    config.finalize_model_grads_func = finalize_model_grads
    config.grad_scale_func = optimizer.scale_loss
    for model_chunk in model:
        model_chunk.eval()
    try:
        for label, (key, mask_key) in coefficient_keys.items():
            group_norm_squares = [
                (
                    norm := backward_norm(
                        subset(analysis_data, start, end), key, group_size, mask_key
                    )
                )
                ** 2
                for start, end in group_ranges
            ]
            result[f"mean_gradient_norm_{label}"] = backward_norm(
                selected, key, selected_samples, mask_key
            )
            result[f"microbatch_gradient_norm_sq_{label}"] = group_norm_squares
    finally:
        config.finalize_model_grads_func = original_finalize
        config.grad_scale_func = original_grad_scale
        for model_chunk, was_training in zip(model, original_modes, strict=True):
            model_chunk.train(was_training)
        for model_chunk in model:
            model_chunk.zero_grad_buffer()
        optimizer.zero_grad()
    return result


def train_one_step(
    args: Namespace,
    rollout_id: int,
    step_id: int,
    data_iterator: Sequence[DataIterator],
    model: Sequence[DDP],
    optimizer: MegatronOptimizer,
    opt_param_scheduler: OptimizerParamScheduler,
    num_microbatches: int,
) -> tuple[dict[str, float], float]:
    """Execute a single pipeline-parallel training step.

    Runs forward/backward over ``num_microbatches``, applies optimizer step and
    one scheduler step when gradients are valid.

    Args:
        args (Namespace): Runtime arguments.
        rollout_id (int): Rollout identifier.
        step_id (int): Step index within the current rollout.
        data_iterator (Sequence[DataIterator]): Iterable(s) yielding training batches.
        model (Sequence[DDP]): Sequence of DDP-wrapped model chunks.
        optimizer (MegatronOptimizer): Optimizer instance.
        opt_param_scheduler (OptimizerParamScheduler): LR/WD scheduler.
        num_microbatches (int): Number of microbatches to process.

    Returns:
        tuple[dict[str, float], float]: Reduced loss dictionary (last stage only)
        and gradient norm for logging.
    """
    args = get_args()

    # Set grad to zero.
    for model_chunk in model:
        model_chunk.zero_grad_buffer()
    optimizer.zero_grad()

    if args.custom_megatron_before_train_step_hook_path:
        from slime.utils.misc import load_function

        custom_before_train_step_hook = load_function(
            args.custom_megatron_before_train_step_hook_path)
        custom_before_train_step_hook(
            args, rollout_id, step_id, model, optimizer, opt_param_scheduler)

    vapo_loss_normalizers = None
    if args.loss_type == "policy_loss" and (
        getattr(args, "ppo_token_level_loss", False) or getattr(args, "use_positive_nll_loss", False)
    ):
        vapo_loss_normalizers = get_vapo_loss_normalizers(args, data_iterator[0], num_microbatches)

    def forward_step(data_iterator: DataIterator, model: GPTModel, return_schedule_plan: bool = False) -> tuple[
        torch.Tensor,
        Callable[[torch.Tensor], tuple[torch.Tensor,
                                       int, dict[str, torch.Tensor | list[str]]]],
    ]:
        """Forward step used by Megatron's pipeline engine during training.

        Args:
            data_iterator (DataIterator): Input data iterator.
            model (GPTModel): The GPT model chunk to execute.

        Returns:
            tuple[torch.Tensor, Callable[[torch.Tensor], tuple[torch.Tensor, int, dict[str, torch.Tensor | list[str]]]]]:
            Output tensor(s) and the loss function, which returns
            (loss, num_elems, {"keys": list[str], "values": torch.Tensor}).
        """

        # Get the batch.
        batch = get_batch(
            data_iterator,
            [
                "tokens",
                "multimodal_train_inputs",
                "packed_seq_params",
                "total_lengths",
                "response_lengths",
                "loss_masks",
                "log_probs",
                "ref_log_probs",
                "values",
                "advantages",
                "returns",
                "rewards",
                "positive_nll_rewards",
                "rollout_log_probs",
                "max_seq_lens",
                "teacher_log_probs",
                "sample_weights",
                "hlg_group_success_p",
                "hlg_group_num_correct",
                "hlg_group_size",
                "mc_value_labels",
                "mc_value_label_masks",
                "mc_value_stds",
                "mc_value_ns",
                
            ],
            args.data_pad_size_multiplier,
            args.qkv_format,
            args.allgather_cp,
        )
        batch["rollout_id"] = rollout_id
        batch["step_id"] = step_id
        if vapo_loss_normalizers is not None:
            batch["vapo_loss_normalizers"] = vapo_loss_normalizers

        if os.environ.get("ENABLE_ROUTING_REPLAY", "0") == "1":
            old_stage = os.environ["ROUTING_REPLAY_STAGE"]
            os.environ["ROUTING_REPLAY_STAGE"] = "replay_forward"

        if return_schedule_plan:
            assert not args.enable_mtp_training, "MTP training should not be enabled when using combined 1f1b"
            output_tensor = model.build_schedule_plan(
                input_ids=batch["tokens"],
                position_ids=None,
                attention_mask=None,
                labels=None,
                packed_seq_params=batch["packed_seq_params"],
                loss_mask=batch["full_loss_masks"],
            )
        else:
            forward_kwargs = {
                "input_ids": batch["tokens"],
                "position_ids": None,
                "attention_mask": None,
                "labels": None,
                "packed_seq_params": batch["packed_seq_params"],
                "loss_mask": batch["full_loss_masks"],
            }

            if args.enable_mtp_training:
                forward_kwargs["mtp_kwargs"] = {"mtp_labels": batch["tokens"]}

            if batch["multimodal_train_inputs"] is not None:
                forward_kwargs.update(batch["multimodal_train_inputs"])

            output_tensor = model(**forward_kwargs)

        if os.environ.get("ENABLE_ROUTING_REPLAY", "0") == "1":
            os.environ["ROUTING_REPLAY_STAGE"] = old_stage

        return output_tensor, partial(loss_function, args, batch, num_microbatches)

    # Forward pass.
    forward_backward_func = get_forward_backward_func()
    losses_reduced = forward_backward_func(
        forward_step_func=forward_step,
        data_iterator=data_iterator,
        model=model,
        num_microbatches=num_microbatches,
        seq_length=args.seq_length,
        micro_batch_size=args.micro_batch_size,
        decoder_seq_length=args.decoder_seq_length,
        forward_only=False,
    )
    valid_step = True
    if not getattr(args, "check_for_nan_in_loss_and_grad", True):
        found_inf_flag = optimizer.prepare_grads()
        if found_inf_flag:
            valid_step = False
        else:
            grad_norm = optimizer.get_grad_norm()
            if isinstance(grad_norm, torch.Tensor):
                valid_step = not (torch.isnan(grad_norm)
                                  or torch.isinf(grad_norm))
            else:
                valid_step = not (math.isnan(grad_norm)
                                  or math.isinf(grad_norm))

    # CI check: verify only MTP parameters have non-zero gradients when truncation happens
    # This check must happen before optimizer.step() as gradients may be modified during step
    if args.ci_test and args.enable_mtp_training:
        from slime.backends.megatron_utils.ci_utils import check_mtp_only_grad

        check_mtp_only_grad(model, step_id)

    if valid_step:
        # Update parameters.
        update_successful, grad_norm, num_zeros_in_grad = optimizer.step()

        # Update learning rate.
        assert update_successful
        opt_param_scheduler.step(increment=args.global_batch_size)

    # release grad
    for model_chunk in model:
        model_chunk.zero_grad_buffer()
    optimizer.zero_grad()

    if mpu.is_pipeline_last_stage(ignore_virtual=True):
        # Average loss across microbatches.
        keys = losses_reduced[0]["keys"]
        values = None
        for x in losses_reduced:
            if values is None:
                values = x["values"]
            else:
                values += x["values"]
        assert len(keys) + 1 == values.numel()
        torch.distributed.all_reduce(
            values, group=mpu.get_data_parallel_group(with_context_parallel=True))

        loss_reduced = {}
        values = values.tolist()
        num_samples_or_tokens = values[0]
        for key, value in zip(keys, values[1:], strict=False):
            loss_reduced[key] = value * \
                mpu.get_context_parallel_world_size() / num_samples_or_tokens
        _finalize_value_explained_variance(loss_reduced)
        return loss_reduced, grad_norm
    return {}, grad_norm

def compute_full_model_per_token_gradient_norms(
    args: Namespace,
    model: Sequence[DDP],
    optimizer: MegatronOptimizer,
    rollout_data: dict,
    positions_by_sample: list[list[int]] | None = None,
) -> list[list[float]]:
    """Compute exact full-model score-gradient norms for selected positions.

    Each selected response position is evaluated with one full-model
    backward.  The objective contains exactly one response sample and one
    selected token, so the returned norm is ``||grad log pi(a_t | s_t)||``;
    it is not an aggregate over tokens or responses.

    Per-sample exactness requires one data-parallel replica.  With more than
    one data-parallel replica, Megatron would reduce different samples across
    ranks before the norm is measured, which would no longer be a single
    anchor's gradient.
    """
    if mpu.get_context_parallel_world_size() != 1:
        raise RuntimeError(
            "per-position full-model gradient analysis requires context_parallel_size == 1"
        )
    if mpu.get_pipeline_model_parallel_world_size() != 1:
        raise RuntimeError(
            "per-position full-model gradient analysis requires pipeline_model_parallel_size == 1"
        )
    dp_size = mpu.get_data_parallel_world_size(with_context_parallel=True)
    if dp_size != 1:
        raise RuntimeError(
            "per-position full-model gradient analysis requires data_parallel_size == 1; "
            f"got {dp_size}"
        )

    response_lengths = [int(length) for length in rollout_data["response_lengths"]]
    total_local_samples = len(response_lengths)
    if positions_by_sample is None:
        positions_by_sample = [
            list(range(response_length)) for response_length in response_lengths
        ]
    if len(positions_by_sample) != total_local_samples:
        raise ValueError(
            "positions_by_sample/response_lengths length mismatch: "
            f"{len(positions_by_sample)} vs {total_local_samples}"
        )

    normalized_positions: list[list[int]] = []
    for sample_index, (positions, response_length) in enumerate(
        zip(positions_by_sample, response_lengths, strict=True)
    ):
        normalized = []
        seen: set[int] = set()
        for position in positions:
            position = int(position)
            if position < 0 or position >= response_length:
                raise ValueError(
                    f"invalid position {position} for sample {sample_index} "
                    f"with response length {response_length}"
                )
            if position not in seen:
                seen.add(position)
                normalized.append(position)
        normalized_positions.append(normalized)

    def one_sample_data(sample_index: int, position: int) -> dict:
        data = {}
        for key, value in rollout_data.items():
            if isinstance(value, (list, tuple)) and len(value) == total_local_samples:
                data[key] = [value[sample_index]]
            else:
                data[key] = value

        response_length = response_lengths[sample_index]
        coefficient = torch.zeros(response_length, dtype=torch.float32)
        mask = torch.zeros(response_length, dtype=torch.float32)
        coefficient[position] = 1.0
        mask[position] = 1.0
        data["analysis_single_token_coefficients"] = [coefficient]
        data["analysis_single_token_masks"] = [mask]
        # This is only the scheduler batch size for the one-sample replay.
        # The reported PPO update scale is applied separately by the caller.
        data["dynamic_global_batch_size"] = 1
        return data

    def backward_one(data: dict) -> float:
        data_iterators, microbatches_by_step = get_data_iterator(args, model, data)
        if len(microbatches_by_step) != 1 or int(microbatches_by_step[0]) <= 0:
            raise RuntimeError(
                "per-position full-model gradient analysis expects one non-empty "
                f"statistical batch, got {microbatches_by_step}"
            )
        for iterator in data_iterators:
            iterator.reset()
        for model_chunk in model:
            model_chunk.zero_grad_buffer()
        optimizer.zero_grad()

        def forward_step(
            data_iterator: DataIterator,
            model_chunk: GPTModel,
            return_schedule_plan: bool = False,
        ):
            batch = get_batch(
                data_iterator,
                [
                    "tokens",
                    "multimodal_train_inputs",
                    "total_lengths",
                    "response_lengths",
                    "loss_masks",
                    "analysis_single_token_coefficients",
                    "analysis_single_token_masks",
                    "max_seq_lens",
                ],
                args.data_pad_size_multiplier,
                args.qkv_format,
                args.allgather_cp,
            )
            forward_kwargs = {
                "input_ids": batch["tokens"],
                "position_ids": None,
                "attention_mask": None,
                "labels": None,
                "packed_seq_params": batch["packed_seq_params"],
                "loss_mask": batch["full_loss_masks"],
            }
            if batch["multimodal_train_inputs"] is not None:
                forward_kwargs.update(batch["multimodal_train_inputs"])
            output_tensor = (
                model_chunk.build_schedule_plan(**forward_kwargs)
                if return_schedule_plan
                else model_chunk(**forward_kwargs)
            )
            return output_tensor, partial(
                full_model_score_gradient_loss,
                args,
                batch,
                "analysis_single_token_coefficients",
                1,
                "analysis_single_token_masks",
            )

        try:
            get_forward_backward_func()(
                forward_step_func=forward_step,
                data_iterator=data_iterators,
                model=model,
                num_microbatches=int(microbatches_by_step[0]),
                seq_length=args.seq_length,
                micro_batch_size=args.micro_batch_size,
                decoder_seq_length=args.decoder_seq_length,
                forward_only=False,
            )
            found_inf = optimizer.prepare_grads()
            if bool(found_inf):
                raise FloatingPointError(
                    "non-finite selected-anchor gradient in full-model analysis"
                )
            norm = optimizer.get_grad_norm()
            return float(norm.item() if isinstance(norm, torch.Tensor) else norm)
        finally:
            for model_chunk in model:
                model_chunk.zero_grad_buffer()
            optimizer.zero_grad()

    config = get_model_config(model[0])
    original_finalize = config.finalize_model_grads_func
    original_grad_scale = config.grad_scale_func
    original_modes = [model_chunk.training for model_chunk in model]
    config.finalize_model_grads_func = finalize_model_grads
    config.grad_scale_func = optimizer.scale_loss
    for model_chunk in model:
        model_chunk.eval()

    norms_by_sample: list[list[float]] = []
    try:
        for sample_index, positions in enumerate(normalized_positions):
            sample_norms = []
            for position in positions:
                sample_norms.append(
                    backward_one(one_sample_data(sample_index, position))
                )
            norms_by_sample.append(sample_norms)
    finally:
        config.finalize_model_grads_func = original_finalize
        config.grad_scale_func = original_grad_scale
        for model_chunk, was_training in zip(model, original_modes, strict=True):
            model_chunk.train(was_training)
        for model_chunk in model:
            model_chunk.zero_grad_buffer()
        optimizer.zero_grad()

    return norms_by_sample


def compute_full_model_per_anchor_gradient_norms(
    args: Namespace,
    model: Sequence[DDP],
    optimizer: MegatronOptimizer,
    rollout_data: dict,
) -> list[list[dict[str, float | int]]]:
    """Compute exact full-actor score-gradient norms only at valid MC anchors.

    The returned entries contain the response-token position and the norm of
    ``grad log pi(token)``.  Critic/MC advantage coefficients are applied by
    the caller so the same expensive score-gradient backward can produce both
    update estimates.
    """
    required = ["mc_value_label_masks", "loss_masks"]
    missing = [key for key in required if rollout_data.get(key) is None]
    if missing:
        raise KeyError(f"per-anchor gradient analysis requires rollout fields: {missing}")

    response_lengths = [int(length) for length in rollout_data["response_lengths"]]
    anchor_positions: list[list[int]] = []
    for sample_index, (mc_mask, loss_mask, response_length) in enumerate(
        zip(
            rollout_data["mc_value_label_masks"],
            rollout_data["loss_masks"],
            response_lengths,
            strict=True,
        )
    ):
        mc_mask_tensor = torch.as_tensor(mc_mask).reshape(-1)
        loss_mask_tensor = torch.as_tensor(loss_mask).reshape(-1)
        if mc_mask_tensor.numel() != response_length or loss_mask_tensor.numel() != response_length:
            raise ValueError(
                "anchor gradient mask length mismatch for sample "
                f"{sample_index}: mc={mc_mask_tensor.numel()}, "
                f"loss={loss_mask_tensor.numel()}, response={response_length}"
            )
        valid = (mc_mask_tensor != 0) & (loss_mask_tensor != 0)
        anchor_positions.append(torch.nonzero(valid, as_tuple=False).reshape(-1).tolist())

    score_norms = compute_full_model_per_token_gradient_norms(
        args,
        model,
        optimizer,
        rollout_data,
        positions_by_sample=anchor_positions,
    )
    return [
        [
            {"position": int(position), "score_gradient_norm": float(norm)}
            for position, norm in zip(positions, norms, strict=True)
        ]
        for positions, norms in zip(anchor_positions, score_norms, strict=True)
    ]

def should_disable_forward_pre_hook(args: Namespace) -> bool:
    """Block forward pre-hook for certain configurations."""
    return args.use_distributed_optimizer and args.overlap_param_gather


def train(
    rollout_id: int,
    model: Sequence[DDP],
    optimizer: MegatronOptimizer,
    opt_param_scheduler: OptimizerParamScheduler,
    data_iterator: Sequence[DataIterator],
    num_microbatches: Sequence[int],
    ) -> None:
    """Run training over a rollout consisting of multiple steps.

    The model is switched to train mode, training hooks are configured, and
    ``train_one_step`` is invoked for each step in the rollout.

    Args:
        rollout_id (int): Rollout identifier.
        model (Sequence[DDP]): Sequence of DDP-wrapped model chunks.
        optimizer (MegatronOptimizer): Optimizer instance.
        opt_param_scheduler (OptimizerParamScheduler): LR/WD scheduler.
        data_iterator (Sequence[DataIterator]): Iterable(s) yielding training batches.
        num_microbatches (Sequence[int]): Microbatches per step in the rollout.
    """
    args = get_args()

    for iterator in data_iterator:
        iterator.reset()

    # Turn on training mode which enables dropout.
    for model_module in model:
        model_module.train()

    # Setup some training config params.
    config = get_model_config(model[0])
    config.grad_scale_func = optimizer.scale_loss
    config.timers = None
    if isinstance(model[0], DDP) and args.overlap_grad_reduce:
        assert config.no_sync_func is None, (
            "When overlap_grad_reduce is True, config.no_sync_func must be None; "
            "a custom no_sync_func is not supported when overlapping grad-reduce"
        )
        config.no_sync_func = [model_chunk.no_sync for model_chunk in model]
        if len(model) == 1:
            config.no_sync_func = config.no_sync_func[0]
        if args.align_grad_reduce:
            config.grad_sync_func = [
                model_chunk.start_grad_sync for model_chunk in model]
            if len(model) == 1:
                config.grad_sync_func = config.grad_sync_func[0]
    if args.overlap_param_gather and args.align_param_gather:
        config.param_sync_func = [
            model_chunk.start_param_sync for model_chunk in model]
        if len(model) == 1:
            config.param_sync_func = config.param_sync_func[0]
    config.finalize_model_grads_func = finalize_model_grads

    pre_hook_enabled = False

    if args.reset_optimizer_states:
        if (
            mpu.get_data_parallel_rank(with_context_parallel=True) == 0
            and mpu.get_tensor_model_parallel_rank() == 0
            and mpu.get_pipeline_model_parallel_rank() == mpu.get_pipeline_model_parallel_world_size() - 1
        ):
            print("Reset optimizer states")
        for chained_optimizer in optimizer.chained_optimizers:
            for group in chained_optimizer.optimizer.param_groups:
                if "step" in group:
                    group["step"] = 0
            for state in chained_optimizer.optimizer.state.values():
                if "step" in state:
                    if isinstance(state["step"], torch.Tensor):
                        state["step"].zero_()
                    else:
                        state["step"] = 0
                if "exp_avg" in state:
                    state["exp_avg"].zero_()
                if "exp_avg_sq" in state:
                    state["exp_avg_sq"].zero_()

    if args.manual_gc:
        # Disable the default garbage collector and perform the collection manually.
        # This is to align the timing of garbage collection across ranks.
        assert args.manual_gc_interval >= 0, "Manual garbage collection interval should be larger than or equal to 0"
        gc.disable()
        gc.collect()

    # Disable forward pre-hook to start training to ensure that errors in checkpoint loading
    # or random initialization don't propagate to all ranks in first all-gather (which is a
    # no-op if things work correctly).
    if should_disable_forward_pre_hook(args):
        disable_forward_pre_hook(model, param_sync=False)
        # Also remove param_sync_func temporarily so that sync calls made in
        # `forward_backward_func` are no-ops.
        param_sync_func = config.param_sync_func
        config.param_sync_func = None
        pre_hook_enabled = False

    num_steps_per_rollout = len(num_microbatches)

    # Run training iterations till done.
    for step_id in range(num_steps_per_rollout):

        # Run training step.
        loss_dict, grad_norm = train_one_step(
            args,
            rollout_id,
            step_id,
            data_iterator,
            model,
            optimizer,
            opt_param_scheduler,
            num_microbatches[step_id],
            )

        if step_id == 0:
            # Enable forward pre-hook after training step has successfully run. All subsequent
            # forward passes will use the forward pre-hook / `param_sync_func` in
            # `forward_backward_func`.
            if should_disable_forward_pre_hook(args):
                enable_forward_pre_hook(model)
                config.param_sync_func = param_sync_func
                pre_hook_enabled = True

        if args.enable_mtp_training:
            from megatron.core.transformer.multi_token_prediction import MTPLossLoggingHelper

            mtp_loss_scale = 1 / num_microbatches[step_id]
            tracker = MTPLossLoggingHelper.tracker
            if "values" in tracker:
                values = tracker["values"]
                if tracker.get("reduce_group") is not None:
                    torch.distributed.all_reduce(
                        values, group=tracker.get("reduce_group"))
                if tracker.get("avg_group") is not None:
                    torch.distributed.all_reduce(
                        values, group=tracker["avg_group"], op=torch.distributed.ReduceOp.AVG)
                # here we assume only one mtp layer
                mtp_losses = (tracker["values"] * mtp_loss_scale).item()
                MTPLossLoggingHelper.clean_loss_in_tracker()

                # CI check: verify MTP loss is within expected bounds
                if args.ci_test:
                    from slime.backends.megatron_utils.ci_utils import check_mtp_loss

                    check_mtp_loss(mtp_losses)

        # per train step log.
        if (
            mpu.get_data_parallel_rank(with_context_parallel=True) == 0
            and mpu.get_tensor_model_parallel_rank() == 0
            and mpu.get_pipeline_model_parallel_rank() == mpu.get_pipeline_model_parallel_world_size() - 1
        ):
            accumulated_step_id = rollout_id * num_steps_per_rollout + step_id
            role = getattr(model[0], "role", "actor")
            role_tag = "" if role == "actor" else f"{role}-"
            log_dict = {
                f"train/{role_tag}{key}": val.mean().item() if isinstance(val, torch.Tensor) else val
                for key, val in loss_dict.items()
            }
            log_dict[f"train/{role_tag}grad_norm"] = grad_norm
            if args.enable_mtp_training:
                log_dict[f"train/{role_tag}mtp_loss"] = mtp_losses

            for param_group_id, param_group in enumerate(optimizer.param_groups):
                log_dict[f"train/{role_tag}lr-pg_{param_group_id}"] = opt_param_scheduler.get_lr(
                    param_group)

            log_dict["train/step"] = accumulated_step_id
            logging_utils.log(args, log_dict, step_key="train/step")

            if args.ci_test and not args.ci_disable_kl_checker:
                if step_id == 0 and "train/ppo_kl" in log_dict and "train/pg_clipfrac" in log_dict:
                    # TODO: figure out why KL is not exactly zero when using PPO loss with KL clipping, and whether this is expected behavior or a bug.
                    assert log_dict["train/ppo_kl"] < 1e-8, f"{log_dict=}"
                if accumulated_step_id == 0 and "train/kl_loss" in log_dict:
                    assert log_dict["train/kl_loss"] < 1e-8, f"{log_dict=}"

            logger.info(f"{role_tag}step {accumulated_step_id}: {log_dict}")

            if args.ci_save_grad_norm is not None:
                ci_save_grad_norm_path = args.ci_save_grad_norm.format(
                    role=role,
                    rollout_id=rollout_id,
                    step_id=step_id,
                )
                torch.save(grad_norm, ci_save_grad_norm_path)
            elif args.ci_load_grad_norm is not None:
                ci_load_grad_norm_path = args.ci_load_grad_norm.format(
                    role=role,
                    rollout_id=rollout_id,
                    step_id=step_id,
                )
                expected_grad_norm = torch.load(ci_load_grad_norm_path)
                assert math.isclose(
                    grad_norm,
                    expected_grad_norm,
                    rel_tol=0.01,
                    abs_tol=0.01,
                ), f"grad norm mismatch: {grad_norm} != {expected_grad_norm}"
    # Close out pre-hooks if using distributed optimizer and overlapped param gather.
    if pre_hook_enabled:
        disable_forward_pre_hook(model)


def save(
    iteration: int, model: Sequence[DDP], optimizer: MegatronOptimizer, opt_param_scheduler: OptimizerParamScheduler
) -> None:
    """Persist a training checkpoint safely with forward hooks disabled.

    Args:
        iteration (int): Current global iteration number.
        model (Sequence[DDP]): Sequence of DDP-wrapped model chunks.
        optimizer (MegatronOptimizer): Optimizer instance.
        opt_param_scheduler (OptimizerParamScheduler): LR/WD scheduler.
    """
    args = get_args()
    if should_disable_forward_pre_hook(args):
        disable_forward_pre_hook(model)
    save_checkpoint(
        iteration,
        model,
        optimizer,
        opt_param_scheduler,
        num_floating_point_operations_so_far=0,
        checkpointing_context=None,
        train_data_iterator=None,
        preprocess_common_state_dict_fn=None,
    )
    if should_disable_forward_pre_hook(args):
        enable_forward_pre_hook(model)


def save_hf_model(args, rollout_id: int, model: Sequence[DDP]) -> None:
    """Save Megatron model in HuggingFace format.

    Args:
        model (Sequence[DDP]): Sequence of DDP-wrapped model chunks.
        rollout_id (int): Rollout ID for path formatting.
    """
    should_log = (
        mpu.get_data_parallel_rank(
            with_context_parallel=True) == 0 and mpu.get_tensor_model_parallel_rank() == 0
    )

    try:
        from megatron.bridge import AutoBridge
        from slime.utils.megatron_bridge_utils import patch_megatron_model

        path = Path(args.save_hf.format(rollout_id=rollout_id))

        if should_log:
            logger.info(f"Saving model in HuggingFace format to {path}")

        bridge = AutoBridge.from_hf_pretrained(
            args.hf_checkpoint, trust_remote_code=True)

        path.mkdir(parents=True, exist_ok=True)

        with patch_megatron_model(model):
            bridge.save_hf_pretrained(
                model,
                path=path,
            )

        if should_log:
            logger.info(f"Successfully saved HuggingFace model to {path}")
    except Exception as e:
        if should_log:
            logger.error(f"Failed to save HuggingFace format: {e}")


def initialize_model_and_optimizer(
    args: Namespace, role: str = "actor"
) -> tuple[list[DDP], MegatronOptimizer, OptimizerParamScheduler, int]:
    """Initialize model(s), optimizer, scheduler, and load from checkpoint.

    Args:
        args (Namespace): Runtime arguments.
        role (str): Logical role of the model (e.g., "actor", "critic").

    Returns:
        tuple[list[DDP], MegatronOptimizer, OptimizerParamScheduler, int]:
            DDP-wrapped model chunks, optimizer, scheduler, and iteration index.
    """

    if torch.version.hip:
        import megatron.core.dist_checkpointing.strategies.filesystem_async as filesystem_async_module
        from slime.utils.rocm_checkpoint_writer import ROCmFileSystemWriterAsync

        filesystem_async_module.FileSystemWriterAsync = ROCmFileSystemWriterAsync
        print("[ROCm] Applied FileSystemWriterAsync patch for HIP compatibility")

    model, optimizer, opt_param_scheduler = setup_model_and_optimizer(
        args, role)
    model[0].role = role
    reinit_critic_output_layer = _critic_output_layer_needs_reinit(
        args, model, role)
    clear_memory()
    iteration, _ = load_checkpoint(
        model,
        optimizer,
        opt_param_scheduler,
        checkpointing_context={},
        skip_load_to_model_and_opt=False,
    )
    if reinit_critic_output_layer:
        zero_init_output_layer = bool(
            getattr(args, "critic_zero_init_output_layer", False))
        logger.warning(
            "Reinitializing critic output layer with %s initialization",
            "zero" if zero_init_output_layer else "normal(mean=0.0, std=0.02)",
        )
        _reinitialize_critic_output_layer(
            model, zero_init=zero_init_output_layer)
        if (args.fp16 or args.bf16) and optimizer is not None:
            optimizer.reload_model_params()
    clear_memory()

    return model, optimizer, opt_param_scheduler, iteration
