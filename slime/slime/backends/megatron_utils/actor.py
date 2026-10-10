import logging
import os
import random
import socket
from argparse import Namespace
from contextlib import nullcontext

import numpy as np
import ray
import torch
import torch.distributed as dist
from megatron.core import mpu
from ray.actor import ActorHandle
from torch_memory_saver import torch_memory_saver
from transformers import AutoConfig, AutoTokenizer

from slime.ray.train_actor import TrainRayActor
from slime.utils import train_dump_utils
from slime.utils.data import process_rollout_data
from slime.utils.distributed_utils import get_gloo_group, init_process_group
from slime.utils.logging_utils import init_tracking
from slime.utils.memory_utils import clear_memory, print_memory
from slime.utils.misc import Box
from slime.utils.reloadable_process_group import destroy_process_groups, monkey_patch_torch_dist, reload_process_groups
from slime.utils.routing_replay import RoutingReplay
from slime.utils.timer import Timer, inverse_timer, timer, with_defer
from slime.utils.types import RolloutBatch

from ...utils.profile_utils import TrainProfiler
from ...utils.tensor_backper import TensorBackuper
from .checkpoint import load_checkpoint
from .cp_utils import slice_log_prob_with_cp, slice_with_cp
from .data import DataIterator, get_data_iterator, log_perf_data, log_rollout_data, sync_actor_critic_data
from .initialize import init, is_megatron_main_rank
from .loss import compute_advantages_and_returns, get_log_probs_and_entropy, get_values
from .sppo import get_sppo_values
from .model import (
    compute_full_model_gradient_statistics,
    compute_full_model_per_anchor_gradient_norms,
    compute_full_model_per_token_gradient_norms,
    compute_output_layer_gradient_statistics,
    forward_only,
    initialize_model_and_optimizer,
    save,
    train,
)
from .update_weight.common import named_params_and_buffers
from .update_weight.update_weight_from_distributed import UpdateWeightFromDistributed
from .update_weight.update_weight_from_tensor import UpdateWeightFromTensor

logging.getLogger("megatron").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


def _per_anchor_gradient_updates(
    args: Namespace,
    rollout_data: RolloutBatch,
    score_gradient_norms: list[list[dict[str, float | int]]],
) -> list[list[dict[str, float | int | bool | None]]]:
    """Attach critic/MC policy-update coefficients to exact anchor score norms.

    ``score_gradient_norms`` already contains the expensive full-model
    ``||grad log pi||`` result for each anchor. PPO pre/post-clip updates differ
    only by a detached scalar coefficient, so both update norms are exact
    without a second backward for the same anchor.
    """
    unsupported = [
        name
        for name, enabled in (
            ("use_opsm", bool(getattr(args, "use_opsm", False))),
            ("get_mismatch_metrics", bool(getattr(args, "get_mismatch_metrics", False))),
            ("use_tis", bool(getattr(args, "use_tis", False))),
            ("use_kl_loss", bool(getattr(args, "use_kl_loss", False))),
            ("use_opd", bool(getattr(args, "use_opd", False))),
            ("normalize_advantages", bool(getattr(args, "normalize_advantages", False))),
            ("nonzero_entropy_coef", float(getattr(args, "entropy_coef", 0.0)) != 0.0),
            (
                "non-ppo advantage estimator",
                getattr(args, "advantage_estimator", "ppo") != "ppo",
            ),
        )
        if enabled
    ]
    if unsupported:
        raise RuntimeError(
            "per-anchor actual PPO gradient export does not yet include "
            f"additional policy coefficient paths: {', '.join(unsupported)}"
        )
    if getattr(args, "custom_pg_loss_reducer_function_path", None) is not None:
        raise RuntimeError(
            "per-anchor actual PPO gradient export requires the default policy-loss reducer"
        )
    if float(getattr(args, "gamma", 1.0)) != 1.0 or float(getattr(args, "lambd", 1.0)) != 1.0:
        raise RuntimeError(
            "per-anchor actual PPO gradient export requires gamma=1 and lambda=1"
        )
    if float(getattr(args, "kl_coef", 0.0)) != 0.0:
        raise RuntimeError(
            "per-anchor actual PPO gradient export requires kl_coef=0"
        )

    required = [
        "returns",
        "values",
        "mc_value_labels",
        "mc_value_label_masks",
        "loss_masks",
        "advantages",
        "_analysis_current_log_probs",
    ]
    missing = [key for key in required if rollout_data.get(key) is None]
    if missing:
        raise KeyError(f"per-anchor gradient export requires rollout fields: {missing}")

    if getattr(args, "use_rollout_logprobs", False):
        old_key = "rollout_log_probs"
    elif rollout_data.get("_analysis_old_log_probs") is not None:
        old_key = "_analysis_old_log_probs"
    else:
        old_key = "log_probs"
    if rollout_data.get(old_key) is None:
        raise KeyError(f"per-anchor gradient export requires {old_key!r}")

    eps_clip = float(getattr(args, "eps_clip", 0.2))
    eps_clip_high = float(getattr(args, "eps_clip_high", eps_clip))
    eps_clip_c = getattr(args, "eps_clip_c", None)
    eps_clip_c = None if eps_clip_c is None else float(eps_clip_c)
    calculate_per_token_loss = bool(getattr(args, "calculate_per_token_loss", False))
    sample_weights = rollout_data.get("sample_weights")
    if sample_weights is None:
        sample_weights = [1.0] * len(rollout_data["response_lengths"])

    rows_by_sample: list[list[dict[str, float | int | bool | None]]] = []
    global_batch_size = int(
        rollout_data.get("dynamic_global_batch_size", args.global_batch_size)
    )
    if global_batch_size <= 0:
        raise ValueError(f"invalid global batch size for anchor gradient export: {global_batch_size}")

    for sample_index, (score_rows, returns, values, mc_values, mc_mask, loss_mask, actual_advantages) in enumerate(
        zip(
            score_gradient_norms,
            rollout_data["returns"],
            rollout_data["values"],
            rollout_data["mc_value_labels"],
            rollout_data["mc_value_label_masks"],
            rollout_data["loss_masks"],
            rollout_data["advantages"],
            strict=True,
        )
    ):
        returns_tensor = torch.as_tensor(returns).detach().float().reshape(-1)
        values_tensor = torch.as_tensor(values, device=returns_tensor.device).detach().float().reshape(-1)
        mc_tensor = torch.as_tensor(mc_values, device=returns_tensor.device).detach().float().reshape(-1)
        mc_mask_tensor = torch.as_tensor(mc_mask, device=returns_tensor.device).detach().float().reshape(-1)
        loss_mask_tensor = torch.as_tensor(loss_mask, device=returns_tensor.device).detach().float().reshape(-1)
        actual_advantage_tensor = torch.as_tensor(
            actual_advantages,
            device=returns_tensor.device,
        ).detach().float().reshape(-1)
        current_log_probs = torch.as_tensor(
            rollout_data["_analysis_current_log_probs"][sample_index],
            device=returns_tensor.device,
        ).detach().float().reshape(-1)
        old_log_probs = torch.as_tensor(
            rollout_data[old_key][sample_index],
            device=returns_tensor.device,
        ).detach().float().reshape(-1)
        expected = returns_tensor.numel()
        if any(
            tensor.numel() != expected
            for tensor in (
                values_tensor,
                mc_tensor,
                mc_mask_tensor,
                loss_mask_tensor,
                actual_advantage_tensor,
                current_log_probs,
                old_log_probs,
            )
        ):
            raise ValueError(f"per-anchor gradient vector length mismatch for sample {sample_index}")

        valid_anchors = (mc_mask_tensor != 0) & (loss_mask_tensor != 0)
        anchor_positions = [int(row["position"]) for row in score_rows]
        if any(not bool(valid_anchors[position]) for position in anchor_positions):
            raise ValueError(f"score-gradient output contains an invalid anchor for sample {sample_index}")
        normalizer = 1.0 if calculate_per_token_loss else max(float(loss_mask_tensor.sum().item()), 1.0)
        sample_weight = float(sample_weights[sample_index])
        ratio = torch.exp(current_log_probs - old_log_probs)
        critic_advantage_raw = returns_tensor - values_tensor
        mc_advantage_raw = returns_tensor - mc_tensor
        # `advantages` is the coefficient actually used by the actor loss.
        # The raw fields expose the unnormalized value errors separately.
        critic_advantage = actual_advantage_tensor
        mc_advantage = mc_advantage_raw

        if calculate_per_token_loss:
            training_scale = sample_weight
        else:
            # The normal Megatron policy loss is a sum of per-response means,
            # followed by the global-batch scaling in _scale_and_pack_megatron_loss.
            # For one response/token this is exactly weight / (tokens * GBS).
            training_scale = sample_weight / (normalizer * global_batch_size)

        def coefficients(advantage: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            pre = ratio * advantage * training_scale
            pg_unclipped = -ratio * advantage
            pg_clipped = -ratio.clamp(1.0 - eps_clip, 1.0 + eps_clip_high) * advantage
            standard_clip = pg_clipped > pg_unclipped
            selected = torch.where(standard_clip, pg_clipped, pg_unclipped)
            dual_clip = torch.zeros_like(standard_clip)
            if eps_clip_c is not None:
                if eps_clip_c <= 1.0:
                    raise ValueError(f"eps_clip_c must be > 1.0, got {eps_clip_c}")
                dual_clip = (advantage < 0) & ((-eps_clip_c * advantage) < selected)
            post = torch.where(
                standard_clip | dual_clip,
                torch.zeros_like(pre),
                pre,
            )
            return pre, post, standard_clip | dual_clip

        critic_pre, critic_post, critic_clipped = coefficients(critic_advantage)
        mc_pre, mc_post, mc_clipped = coefficients(mc_advantage)
        sample_rows: list[dict[str, float | int | bool | None]] = []
        for score_row in score_rows:
            position = int(score_row["position"])
            score_norm = float(score_row["score_gradient_norm"])
            critic_pre_value = float(abs(critic_pre[position].item()) * score_norm)
            mc_pre_value = float(abs(mc_pre[position].item()) * score_norm)
            critic_post_value = float(abs(critic_post[position].item()) * score_norm)
            mc_post_value = float(abs(mc_post[position].item()) * score_norm)

            def cosine(lhs: float, rhs: float) -> float | None:
                if lhs == 0.0 or rhs == 0.0:
                    return None
                return 1.0 if lhs * rhs > 0.0 else -1.0

            sample_rows.append(
                {
                    "position": position,
                    "score_gradient_norm": score_norm,
                    "critic_advantage_raw": float(critic_advantage_raw[position].item()),
                    "critic_advantage": float(critic_advantage[position].item()),
                    "mc_advantage_raw": float(mc_advantage_raw[position].item()),
                    "mc_advantage": float(mc_advantage[position].item()),
                    "ratio": float(ratio[position].item()),
                    "critic_update_norm_preclip": critic_pre_value,
                    "mc_update_norm_preclip": mc_pre_value,
                    "critic_update_norm_postclip": critic_post_value,
                    "mc_update_norm_postclip": mc_post_value,
                    # The unsuffixed fields are the actual post-clip update.
                    "critic_update_norm": critic_post_value,
                    "mc_update_norm": mc_post_value,
                    "critic_mc_direction_cosine_preclip": cosine(
                        float(critic_pre[position].item()), float(mc_pre[position].item())
                    ),
                    "critic_mc_direction_cosine_postclip": cosine(
                        float(critic_post[position].item()), float(mc_post[position].item())
                    ),
                    "critic_mc_direction_cosine": cosine(
                        float(critic_post[position].item()), float(mc_post[position].item())
                    ),
                    "critic_clip_applied": bool(critic_clipped[position].item()),
                    "mc_clip_applied": bool(mc_clipped[position].item()),
                    "normalizer": float(normalizer),
                    "global_batch_size": global_batch_size,
                    "training_gradient_scale": float(training_scale),
                    "sample_weight": sample_weight,
                }
            )
        rows_by_sample.append(sample_rows)
    return rows_by_sample

class MegatronTrainRayActor(TrainRayActor):
    @staticmethod
    def _progress_to_weight(progress_percent: float, min_weight: float, power: float) -> float:
        p = max(0.0, min(100.0, float(progress_percent))) / 100.0
        w_min = max(0.0, min(1.0, float(min_weight)))
        gamma = max(1e-6, float(power))
        return w_min + (1.0 - w_min) * (p**gamma)

    def _build_sample_weights_from_metadata(self, rollout_data: RolloutBatch) -> list[float]:
        metadata_list = rollout_data.get("metadata")
        if metadata_list is None:
            return [1.0] * len(rollout_data["response_lengths"])

        weights: list[float] = []
        for md in metadata_list:
            progress_percent = 100.0
            if isinstance(md, dict):
                raw_progress = md.get("progress_percent", 100.0)
                try:
                    progress_percent = float(raw_progress)
                except (TypeError, ValueError):
                    progress_percent = 100.0
            weights.append(
                self._progress_to_weight(
                    progress_percent,
                    min_weight=self.args.critic_pretrain_reweight_min,
                    power=self.args.critic_pretrain_reweight_power,
                )
            )
        return weights

    @with_defer(lambda: Timer().start("train_wait"))
    def init(
        self,
        args: Namespace,
        role: str,
        with_ref: bool = False,
        with_opd_teacher: bool = False,
    ) -> int | None:
        if args.debug_rollout_only:
            self.args = args
            return 0

        monkey_patch_torch_dist()
        super().init(args, role, with_ref, with_opd_teacher)

        init(args)

        if is_megatron_main_rank():
            init_tracking(args, primary=False)

        self.prof = TrainProfiler(args)

        # read config and tokenizer serialized to prevent concurrent writing bug.
        for i in range(args.num_gpus_per_node):
            if i == dist.get_rank() % args.num_gpus_per_node:
                self.hf_config = AutoConfig.from_pretrained(
                    args.hf_checkpoint, trust_remote_code=True)
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.args.hf_checkpoint, trust_remote_code=True)
            dist.barrier(group=get_gloo_group())

        self.train_parallel_config = {
            "dp_size": mpu.get_data_parallel_world_size(with_context_parallel=False),
        }
        dist.barrier(group=get_gloo_group())

        if args.offload_train:
            if (x := args.train_memory_margin_bytes) > 0:
                logger.info(
                    f"Set torch_memory_saver.memory_margin_bytes to {x}")
                torch_memory_saver.memory_margin_bytes = x

        if role == "critic":
            self.args.load = self.args.critic_load
            self.args.save = self.args.critic_save
            self.args.lr = self.args.critic_lr
            self.args.lr_warmup_iters = self.args.critic_lr_warmup_iters
            if (critic_ckpt_step := getattr(self.args, "critic_ckpt_step", None)) is not None:
                self.args.ckpt_step = critic_ckpt_step

        (self.model, self.optimizer, self.opt_param_scheduler, loaded_rollout_id) = initialize_model_and_optimizer(
            args, role
        )

        start_rollout_id = loaded_rollout_id + 1
        self._train_memory_paused = False

        self.rollout_data_postprocess = None
        if self.args.rollout_data_postprocess_path is not None:
            from slime.utils.misc import load_function

            self.rollout_data_postprocess = load_function(
                self.args.rollout_data_postprocess_path)

        if role == "critic":
            if self.args.offload_train:
                self.sleep()
            return start_rollout_id

        self.weights_backuper = TensorBackuper.create(
            source_getter=lambda: named_params_and_buffers(
                self.args,
                self.model,
                convert_to_global_name=args.megatron_to_hf_mode == "raw",
                translate_gpu_to_cpu=not self.args.enable_weights_backuper,
            ),
            single_tag=None if args.enable_weights_backuper else "actor",
        )
        self._active_model_tag: str | None = "actor"
        logger.info(
            "[%s init] rank=%s backing up actor weights",
            role,
            dist.get_rank(),
        )
        self.weights_backuper.backup("actor")
        dist.barrier(group=get_gloo_group())

        if with_ref:
            logger.info(
                "[%s init] rank=%s loading ref checkpoint from %s",
                role,
                dist.get_rank(),
                args.ref_load,
            )
            self.load_other_checkpoint("ref", args.ref_load)
            dist.barrier(group=get_gloo_group())
            logger.info(
                "[%s init] rank=%s finished ref checkpoint load",
                role,
                dist.get_rank(),
            )

        # Load teacher model for Megatron-based on-policy distillation
        if with_opd_teacher:
            self.load_other_checkpoint("teacher", args.opd_teacher_load)

        if self.args.keep_old_actor:
            # Load old_actor checkpoint
            self.load_other_checkpoint("old_actor", args.load)
            # Create rollout_actor as a copy of current actor
            if args.update_weights_interval == 1:
                self.weights_backuper.backup("rollout_actor")

        if self.args.vocab_size is None:
            # Prefer HF config vocab_size (which may include model-native padding)
            # over tokenizer vocab_size (which may be smaller, e.g. GPT-OSS).
            hf_vocab = getattr(self.hf_config, "vocab_size", None)
            self.args.vocab_size = hf_vocab if hf_vocab is not None else self.tokenizer.vocab_size

        update_weight_cls = UpdateWeightFromTensor if self.args.colocate else UpdateWeightFromDistributed
        self.weight_updater = update_weight_cls(
            self.args,
            self.model,
            weights_getter=lambda: self.weights_backuper.get("actor"),
            model_name=type(self.hf_config).__name__.lower(
            ) if self.args.model_name is None else self.args.model_name,
            quantization_config=getattr(
                self.hf_config, "quantization_config", None),
        )

        # empty cache after initialization
        clear_memory()

        if self.args.offload_train:
            # recover to actor in the end.
            dist.barrier(group=get_gloo_group())
            logger.info(
                "[%s init] rank=%s restoring actor before initial offload",
                role,
                dist.get_rank(),
            )
            self._switch_model("actor")
            dist.barrier(group=get_gloo_group())
            self.sleep()

        self.rollout_engines = None

        self.prof.on_init_end()

        return start_rollout_id

    @timer
    def sleep(self) -> None:
        assert self.args.offload_train

        clear_memory(clear_host_memory=True)
        print_memory("before offload model")
        dist.barrier(group=get_gloo_group())
        logger.info(
            "rank=%s destroying reloadable process groups before offload", dist.get_rank())
        destroy_process_groups()

        logger.info("rank=%s pausing torch_memory_saver", dist.get_rank())
        torch_memory_saver.pause()
        self._train_memory_paused = True

        print_memory("after offload model")

    @timer
    def wake_up(self) -> None:
        assert self.args.offload_train
        print_memory("before wake_up model")

        if self._train_memory_paused:
            torch_memory_saver.resume()
            self._train_memory_paused = False

        clear_memory()
        reload_process_groups()
        print_memory("after wake_up model")

    def _get_rollout_data(self, rollout_data_ref: Box) -> RolloutBatch:
        # Fetch data through ray on CPU, not sure if this will be performance bottleneck.
        # Both first pp stage and the last pp stage will receive the data.
        rollout_data = process_rollout_data(
            self.args,
            rollout_data_ref,
            mpu.get_data_parallel_rank(with_context_parallel=False),
            mpu.get_data_parallel_world_size(with_context_parallel=False),
        )
        if self.args.critic_pretrain_reweight:
            rollout_data["sample_weights"] = self._build_sample_weights_from_metadata(
                rollout_data)
            if mpu.get_tensor_model_parallel_rank() == 0 and mpu.is_pipeline_last_stage():
                sample_weights = rollout_data["sample_weights"]
                metadata_list = rollout_data.get("metadata", [])
                progress_values = []
                for md in metadata_list:
                    if isinstance(md, dict):
                        try:
                            progress_values.append(
                                float(md.get("progress_percent", 100.0)))
                        except (TypeError, ValueError):
                            continue
                if sample_weights:
                    logger.info(
                        "[critic_pretrain_reweight] weight_mean=%.4f weight_min=%.4f weight_max=%.4f samples=%d",
                        float(np.mean(sample_weights)),
                        float(np.min(sample_weights)),
                        float(np.max(sample_weights)),
                        len(sample_weights),
                    )
                if progress_values:
                    logger.info(
                        "[critic_pretrain_reweight] progress_mean=%.2f progress_min=%.2f progress_max=%.2f n_progress=%d",
                        float(np.mean(progress_values)),
                        float(np.min(progress_values)),
                        float(np.max(progress_values)),
                        len(progress_values),
                    )

        rollout_data["tokens"] = [
            torch.tensor(t, dtype=torch.long, device=torch.cuda.current_device()) for t in rollout_data["tokens"]
        ]
        rollout_data["loss_masks"] = [
            torch.tensor(t, dtype=torch.int, device=torch.cuda.current_device()) for t in rollout_data["loss_masks"]
        ]
        if "multimodal_train_inputs" in rollout_data:
            # Move multimodal training tensors to GPU in advance
            rollout_data["multimodal_train_inputs"] = [
                (
                    {
                        key: (
                            torch.from_numpy(v.copy()).to(
                                device=torch.cuda.current_device())
                            if isinstance(v, np.ndarray)
                            else v.to(device=torch.cuda.current_device())
                        )
                        for key, v in mm_dict.items()
                    }
                    if mm_dict is not None
                    else None
                )
                for mm_dict in rollout_data["multimodal_train_inputs"]
            ]

        if self.args.qkv_format == "bshd":
            # TODO: micro-batch wise dynamic, possibly move to @data.py:get_data_iterator
            max_seq_len = max(rollout_data["total_lengths"])

            # pad to reduce memory fragmentation and maybe make the computation faster
            pad_size = mpu.get_tensor_model_parallel_world_size() * \
                self.args.data_pad_size_multiplier
            max_seq_len = (max_seq_len + pad_size - 1) // pad_size * pad_size

            rollout_data["max_seq_lens"] = [
                max_seq_len] * len(rollout_data["tokens"])

        if "mc_value_label_masks" in rollout_data:
            rollout_data["mc_value_label_masks"] = [
                torch.tensor(t, dtype=torch.int,
                             device=torch.cuda.current_device())
                for t in rollout_data["mc_value_label_masks"]
            ]

        if "mc_value_labels" in rollout_data:
            rollout_data["mc_value_labels"] = [
                torch.tensor(
                    slice_log_prob_with_cp(
                        labels,
                        total_length,
                        response_length,
                        self.args.qkv_format,
                        rollout_data["max_seq_lens"][i] if self.args.qkv_format == "bshd" else None,
                    ),
                    device=torch.cuda.current_device(),
                    dtype=torch.float32,
                )
                for i, (labels, total_length, response_length) in enumerate(
                    zip(
                        rollout_data["mc_value_labels"],
                        rollout_data["total_lengths"],
                        rollout_data["response_lengths"],
                        strict=False,
                    )
                )
            ]

        for key in ["mc_value_stds", "mc_value_ns"]:
            if key not in rollout_data:
                continue
            rollout_data[key] = [
                torch.tensor(
                    slice_log_prob_with_cp(
                        values,
                        total_length,
                        response_length,
                        self.args.qkv_format,
                        rollout_data["max_seq_lens"][i] if self.args.qkv_format == "bshd" else None,
                    ),
                    device=torch.cuda.current_device(),
                    dtype=torch.float32,
                )
                for i, (values, total_length, response_length) in enumerate(
                    zip(
                        rollout_data[key],
                        rollout_data["total_lengths"],
                        rollout_data["response_lengths"],
                        strict=False,
                    )
                )
            ]

        for key in ["rollout_log_probs", "teacher_log_probs"]:
            if key not in rollout_data:
                continue
            rollout_data[key] = [
                torch.tensor(
                    slice_log_prob_with_cp(
                        log_prob,
                        total_length,
                        response_length,
                        self.args.qkv_format,
                        rollout_data["max_seq_lens"][i] if self.args.qkv_format == "bshd" else None,
                    ),
                    device=torch.cuda.current_device(),
                    dtype=torch.float32,
                )
                for i, (log_prob, total_length, response_length) in enumerate(
                    zip(
                        rollout_data[key],
                        rollout_data["total_lengths"],
                        rollout_data["response_lengths"],
                        strict=False,
                    )
                )
            ]
        if "rollout_routed_experts" in rollout_data:
            rollout_data["rollout_routed_experts"] = [
                torch.from_numpy(r) for r in rollout_data["rollout_routed_experts"]
            ]
        return rollout_data

    def _switch_model(self, target_tag: str) -> None:
        if target_tag not in self.weights_backuper.backup_tags:
            raise ValueError(
                f"Cannot switch to unknown model tag: {target_tag}")
        self.weights_backuper.restore(target_tag)
        self._active_model_tag = target_tag

    def fill_routing_replay(self, data_iterator, num_microbatches, rollout_data):
        if "rollout_routed_experts" not in rollout_data:
            raise ValueError(
                "rollout_routed_experts is required in rollout_data when use_rollout_routing_replay is set."
            )

        from megatron.core.transformer.transformer_block import get_num_layers_to_build
        from megatron.core.transformer.transformer_layer import get_transformer_layer_offset

        from slime.utils.routing_replay import RoutingReplay

        for iterator in data_iterator:
            iterator.reset()

        tp_rank = mpu.get_tensor_model_parallel_rank()
        tp_size = mpu.get_tensor_model_parallel_world_size()

        def pad_func(experts, pad):
            _, num_layers, topk = experts.shape
            pad = (
                torch.arange(
                    pad * num_layers * topk,
                    device=experts.device,
                    dtype=experts.dtype,
                ).reshape((pad, num_layers, topk))
                % self.args.num_experts
            )
            return torch.cat([experts, pad], dim=0)

        for _ in range(sum(num_microbatches)):
            batch = data_iterator[0].get_next(
                ["rollout_routed_experts", "tokens"])
            rollout_routed_experts = batch["rollout_routed_experts"]
            tokens = batch["tokens"]
            assert len(rollout_routed_experts) == len(tokens)
            for a, b in zip(rollout_routed_experts, tokens, strict=False):
                assert a.shape[0] == b.shape[0] - 1, f"{a.shape}, {b.shape}"

            # We need to pad the experts to the last token. We won't calculate loss on this token so this should be fine.
            # TODO: fuse this padding with the following slice_with_cp to reduce memory copy.
            rollout_routed_experts = [
                pad_func(r, 1) for r in rollout_routed_experts]
            # TODO: maybe extract a common process function for here and get_batch?
            rollout_routed_experts = [slice_with_cp(
                r, pad_func) for r in rollout_routed_experts]
            rollout_routed_experts = torch.cat(rollout_routed_experts, dim=0)
            pad_size = mpu.get_tensor_model_parallel_world_size() * \
                self.args.data_pad_size_multiplier
            pad = (pad_size - rollout_routed_experts.size(0) %
                   pad_size) % pad_size
            if pad != 0:
                rollout_routed_experts = pad_func(rollout_routed_experts, pad)

            if self.args.sequence_parallel:
                seqlen = rollout_routed_experts.size(0)
                assert seqlen % tp_size == 0
                start, end = seqlen // tp_size * \
                    tp_rank, seqlen // tp_size * (tp_rank + 1)
                rollout_routed_experts = rollout_routed_experts[start:end]

            routing_replay_offset = 0
            for vp_stage, model in enumerate(self.model):
                config = model.module.config
                num_layers_to_build = get_num_layers_to_build(
                    config, vp_stage=vp_stage)
                offset = get_transformer_layer_offset(
                    config, vp_stage=vp_stage)
                for layer_id in range(offset, offset + num_layers_to_build):
                    # skip dense layer
                    if isinstance(config.moe_layer_freq, int):
                        if layer_id % config.moe_layer_freq != 0:
                            continue
                    elif isinstance(config.moe_layer_freq, list):
                        assert len(config.moe_layer_freq) == config.num_layers
                        if config.moe_layer_freq[layer_id] == 0:
                            continue
                    layer_routed_experts = rollout_routed_experts[:, layer_id]
                    RoutingReplay.all_routing_replays[routing_replay_offset].record(
                        layer_routed_experts)
                    routing_replay_offset += 1
            assert routing_replay_offset == len(
                RoutingReplay.all_routing_replays)

        del rollout_data["rollout_routed_experts"]

        for iterator in data_iterator:
            iterator.reset()

    def compute_log_prob(
        self,
        data_iterator: list[DataIterator],
        num_microbatches: list[int],
        store_prefix: str = "",
    ) -> dict[str, list[torch.Tensor]]:

        with timer(f"{store_prefix}log_probs"):
            return forward_only(
                get_log_probs_and_entropy,
                self.args,
                self.model,
                data_iterator,
                num_microbatches,
                store_prefix=store_prefix,
            )

    def train(self, rollout_id: int, rollout_data_ref: Box) -> None:
        if self.args.debug_rollout_only:
            return

        if self.args.offload_train:
            self.wake_up()

        with timer("data_preprocess"):
            rollout_data = self._get_rollout_data(rollout_data_ref)

        if self.role == "critic":
            return self.train_critic(rollout_id, rollout_data)
        else:
            return self.train_actor(rollout_id, rollout_data)

    def analysis_export(self, rollout_id: int, rollout_data_ref: Box) -> None:
        if self.args.debug_rollout_only:
            return

        if self.args.offload_train:
            self.wake_up()

        try:
            with timer("data_preprocess"):
                rollout_data = self._get_rollout_data(rollout_data_ref)

            if self.role == "critic":
                self.analysis_export_critic(rollout_id, rollout_data)
            else:
                self.analysis_export_actor(rollout_id, rollout_data)
        finally:
            if self.args.offload_train:
                self.sleep()

    def _prepare_hlg_debug_distribution_mask(self, rollout_data: RolloutBatch) -> None:
        num_samples = int(
            getattr(self.args, "hlg_debug_distribution_samples", 0))
        if num_samples <= 0:
            return

        mask = [False] * len(rollout_data["response_lengths"])
        cp_world_size = mpu.get_context_parallel_world_size()
        if cp_world_size != 1 and is_megatron_main_rank():
            logger.warning(
                "Skip HLG value distribution debug snapshots because context_parallel_size=%d; "
                "sampled full-response distributions currently require CP size 1.",
                cp_world_size,
            )
        should_capture = (
            getattr(self.args, "critic_value_loss_type", "mse") == "hl_gauss"
            and mpu.get_data_parallel_rank(with_context_parallel=False) == 0
            and cp_world_size == 1
            and mpu.get_context_parallel_rank() == 0
            and mpu.is_pipeline_last_stage()
        )
        if should_capture:
            for i in range(min(num_samples, len(mask))):
                mask[i] = True
        rollout_data["hlg_debug_distribution_mask"] = mask

    def train_critic(self, rollout_id: int, rollout_data: RolloutBatch) -> None:
        self._prepare_hlg_debug_distribution_mask(rollout_data)

        # Create data iterator for log_probs and train.
        data_iterator, num_microbatches = get_data_iterator(
            self.args, self.model, rollout_data)
        critic_value_fn = get_sppo_values if self.args.advantage_estimator == "sppo" else get_values
        rollout_data.update(
            forward_only(
                critic_value_fn,
                self.args,
                self.model,
                data_iterator,
                num_microbatches,
            )
        )

        if rollout_id >= self.args.num_critic_only_steps and not self.args.critic_train_only:
            sync_actor_critic_data(
                self.args, rollout_data, self._actor_critic_groups)

        compute_advantages_and_returns(self.args, rollout_data)
        train_dump_utils.save_hlg_debug_value_distribution(
            self.args, rollout_id=rollout_id, rollout_data=rollout_data)

        self.args.loss_type = (
            "sppo_value_loss"
            if self.args.advantage_estimator == "sppo"
            else "value_loss"
        )
        train(
            rollout_id,
            self.model,
            self.optimizer,
            self.opt_param_scheduler,
            data_iterator,
            num_microbatches,
        )

    def analysis_export_critic(self, rollout_id: int, rollout_data: RolloutBatch) -> None:
        self._prepare_hlg_debug_distribution_mask(rollout_data)
        data_iterator, num_microbatches = get_data_iterator(
            self.args, self.model, rollout_data)
        critic_value_fn = get_sppo_values if self.args.advantage_estimator == "sppo" else get_values
        rollout_data.update(
            forward_only(
                critic_value_fn,
                self.args,
                self.model,
                data_iterator,
                num_microbatches,
            )
        )
        if not self.args.critic_train_only:
            sync_actor_critic_data(
                self.args, rollout_data, self._actor_critic_groups)
        elif self.rollout_data_postprocess is not None:
            # In critic-only analysis there is no actor process to receive the
            # synchronized values and invoke the export hook.
            self.rollout_data_postprocess(
                self.args, rollout_id, rollout_data)

    def train_actor(self, rollout_id: int, rollout_data: RolloutBatch) -> None:
        # Create data iterator for log_probs and train.
        data_iterator, num_microbatches = get_data_iterator(
            self.args, self.model, rollout_data)

        if self.args.use_rollout_routing_replay:
            self.fill_routing_replay(
                data_iterator, num_microbatches, rollout_data)

        with inverse_timer("train_wait"), timer("train"):
            if self.args.compute_advantages_and_returns:
                if (
                    "ref" in self.weights_backuper.backup_tags
                    and (self.args.kl_coef != 0 or self.args.use_kl_loss)
                ):
                    if self.args.use_routing_replay:
                        os.environ["ROUTING_REPLAY_STAGE"] = "fallthrough"
                    self._switch_model("ref")
                    rollout_data.update(
                        self.compute_log_prob(
                            data_iterator,
                            num_microbatches,
                            store_prefix="ref_",
                        )
                    )

                # Forward teacher model to get teacher_log_probs for Megatron-based OPD
                if "teacher" in self.weights_backuper.backup_tags:
                    if self.args.use_routing_replay:
                        os.environ["ROUTING_REPLAY_STAGE"] = "fallthrough"
                    self._switch_model("teacher")
                    rollout_data.update(
                        self.compute_log_prob(
                            data_iterator,
                            num_microbatches,
                            store_prefix="teacher_",
                        )
                    )

                self._switch_model(
                    "old_actor" if self.args.keep_old_actor else "actor")
                if not self.args.use_rollout_logprobs or self.args.get_mismatch_metrics:
                    if self.args.use_routing_replay:
                        if self.args.use_rollout_routing_replay:
                            os.environ["ROUTING_REPLAY_STAGE"] = "replay_forward"
                        else:
                            os.environ["ROUTING_REPLAY_STAGE"] = "record"
                    rollout_data.update(
                        self.compute_log_prob(
                            data_iterator,
                            num_microbatches,
                            store_prefix="",
                        )
                    )
                    if self.args.use_rollout_routing_replay:
                        RoutingReplay.clear_all_forward()

                if self.args.use_critic:
                    sync_actor_critic_data(
                        self.args,
                        rollout_data,
                        self._actor_critic_groups,
                    )
                if self._active_model_tag != "actor":
                    self._switch_model("actor")

                # Calculate adv and returns. Need to performed before training (instead of on the fly),
                # because we may need normalize the whole rollout.
                compute_advantages_and_returns(self.args, rollout_data)

            if self.rollout_data_postprocess is not None:
                self.rollout_data_postprocess(
                    self.args, rollout_id, rollout_data)

            log_rollout_data(
                rollout_id,
                self.args,
                rollout_data,
            )

            # Train
            if self.args.use_routing_replay:
                os.environ["ROUTING_REPLAY_STAGE"] = "replay_backward"
            with timer("actor_train"):
                train(
                    rollout_id,
                    self.model,
                    self.optimizer,
                    self.opt_param_scheduler,
                    data_iterator,
                    num_microbatches,
                    )

            self.prof.step(rollout_id=rollout_id)

        train_dump_utils.save_debug_train_data(
            self.args, rollout_id=rollout_id, rollout_data=rollout_data)

        if self.args.use_routing_replay:
            RoutingReplay.clear_all()

        # update the cpu actor weight to the latest model
        self.weights_backuper.backup("actor")

        # Update ref model if needed
        if (
            self.args.ref_update_interval is not None
            and (rollout_id + 1) % self.args.ref_update_interval == 0
            and "ref" in self.weights_backuper.backup_tags
        ):
            with timer("ref_model_update"):
                if is_megatron_main_rank():
                    logger.info(
                        f"Updating ref model at rollout_id {rollout_id}")
                self.weights_backuper.backup("ref")

        log_perf_data(rollout_id, self.args)

    def analysis_export_actor(self, rollout_id: int, rollout_data: RolloutBatch) -> None:
        data_iterator, num_microbatches = get_data_iterator(
            self.args, self.model, rollout_data)

        if (
            "ref" in self.weights_backuper.backup_tags
            and (self.args.kl_coef != 0 or self.args.use_kl_loss)
        ):
            self._switch_model("ref")
            rollout_data.update(
                self.compute_log_prob(
                    data_iterator,
                    num_microbatches,
                    store_prefix="ref_",
                )
            )

        if "teacher" in self.weights_backuper.backup_tags:
            self._switch_model("teacher")
            rollout_data.update(
                self.compute_log_prob(
                    data_iterator,
                    num_microbatches,
                    store_prefix="teacher_",
                )
            )

        # A replay dump may already contain rollout-time policy log-probs.
        # Preserve them before compute_log_prob overwrites `log_probs` with
        # the checkpoint's current-policy values.
        rollout_data["_analysis_old_log_probs"] = rollout_data.get("log_probs")
        if getattr(self.args, "analysis_old_actor_checkpoint", None):
            if not getattr(self.args, "analysis_clipped_gradient", False):
                raise RuntimeError("--analysis-old-actor-checkpoint requires --analysis-clipped-gradient")
            self.load_other_checkpoint("analysis_old_actor", self.args.analysis_old_actor_checkpoint)
            for iterator in data_iterator:
                iterator.reset()
            old_policy = self.compute_log_prob(
                data_iterator,
                num_microbatches,
                store_prefix="analysis_old_",
            )
            rollout_data["_analysis_old_log_probs"] = old_policy["analysis_old_log_probs"]
            rollout_data["_analysis_old_log_probs_source"] = "analysis_old_actor_checkpoint"

        self._switch_model(
            "old_actor" if self.args.keep_old_actor else "actor")
        rollout_data.update(
            self.compute_log_prob(
                data_iterator,
                num_microbatches,
                store_prefix="",
            )
        )
        # Preserve the actor log-probabilities before advantage processing.
        # They are needed to form the exact PPO ratio at each MC anchor.
        rollout_data["_analysis_current_log_probs"] = rollout_data.get("log_probs")
        if getattr(self.args, "analysis_clipped_gradient", False) and self.args.keep_old_actor:
            # Training computes old-policy log-probs above, but backwards through
            # the actor.  Use that actor for the clipped-gradient ratio as well.
            if rollout_data.get("_analysis_old_log_probs") is None:
                rollout_data["_analysis_old_log_probs"] = rollout_data["_analysis_current_log_probs"]
            self._switch_model("actor")
            for iterator in data_iterator:
                iterator.reset()
            current_actor = self.compute_log_prob(
                data_iterator, num_microbatches, store_prefix="analysis_actor_"
            )
            rollout_data["_analysis_current_log_probs"] = current_actor["analysis_actor_log_probs"]

        if self.args.use_critic:
            sync_actor_critic_data(
                self.args,
                rollout_data,
                self._actor_critic_groups,
            )
        if self._active_model_tag != "actor":
            self._switch_model("actor")

        compute_advantages_and_returns(self.args, rollout_data)
        
        if getattr(self.args, "analysis_per_anchor_gradient", False):
            with timer("full_model_per_anchor_gradient_analysis"):
                score_gradient_norms = compute_full_model_per_anchor_gradient_norms(
                    self.args,
                    self.model,
                    self.optimizer,
                    rollout_data,
                )
                rollout_data["per_anchor_gradient_updates"] = _per_anchor_gradient_updates(
                    self.args,
                    rollout_data,
                    score_gradient_norms,
                )
        elif getattr(self.args, "analysis_per_token_gradient", False):
            with timer("full_model_per_token_gradient_analysis"):
                rollout_data["per_token_gradient_norms"] = compute_full_model_per_token_gradient_norms(
                    self.args,
                    self.model,
                    self.optimizer,
                    rollout_data,
                )
        if getattr(self.args, "analysis_full_gradient", False):
            with timer("full_model_gradient_analysis"):
                rollout_data["output_layer_gradient_analysis"] = compute_output_layer_gradient_statistics(
                    self.args,
                    self.model,
                    self.optimizer,
                    rollout_data,
                )
                rollout_data["full_gradient_analysis"] = compute_full_model_gradient_statistics(
                    self.args,
                    self.model,
                    self.optimizer,
                    rollout_data,
                )

        if self.rollout_data_postprocess is not None:
            self.rollout_data_postprocess(
                self.args, rollout_id, rollout_data)

        log_rollout_data(
            rollout_id,
            self.args,
            rollout_data,
        )
        train_dump_utils.save_debug_train_data(
            self.args, rollout_id=rollout_id, rollout_data=rollout_data)

    @timer
    def save_model(
        self,
        rollout_id: int,
        force_sync: bool = False,
        save_dir: str | None = None,
    ) -> None:
        if self.args.debug_rollout_only:
            return

        # torch dist may trigger nccl communication during saving.
        if self.args.offload_train:
            reload_process_groups()

        if self.args.async_save:
            from megatron.training.async_utils import maybe_finalize_async_save

            maybe_finalize_async_save(blocking=True)

        original_save_dir = self.args.save
        if save_dir is not None:
            self.args.save = save_dir
        try:
            save(rollout_id, self.model, self.optimizer, self.opt_param_scheduler)
        finally:
            self.args.save = original_save_dir

        if force_sync and self.args.async_save:
            maybe_finalize_async_save(blocking=True)

        if self.args.save_hf is not None and self.role == "actor":
            from slime.backends.megatron_utils.model import save_hf_model

            save_hf_model(self.args, rollout_id, self.model)

        if self.args.offload_train:
            destroy_process_groups()

    @timer
    def update_weights(self) -> None:
        if self.args.debug_train_only or self.args.debug_rollout_only:
            return

        if self.args.use_fault_tolerance:
            if dist.get_rank() == 0:
                ray.get(self.rollout_manager.recover_updatable_engines.remote())
            dist.barrier(group=get_gloo_group())

        rollout_engines, rollout_engine_lock, num_new_engines, engine_gpu_counts, engine_gpu_offsets = ray.get(
            self.rollout_manager.get_updatable_engines_and_lock.remote()
        )

        if self.args.offload_train:
            reload_process_groups()

        if num_new_engines > 0:
            self.weight_updater.connect_rollout_engines(
                rollout_engines,
                rollout_engine_lock,
                engine_gpu_counts=engine_gpu_counts,
                engine_gpu_offsets=engine_gpu_offsets,
            )
            dist.barrier(group=get_gloo_group())
            if dist.get_rank() == 0:
                ray.get(
                    self.rollout_manager.clear_updatable_num_new_engines.remote())

        with torch_memory_saver.disable() if self.args.offload_train else nullcontext():
            print_memory("before update_weights")
            self.weight_updater.update_weights()
            print_memory("after update_weights")

            if self.args.ci_test and len(rollout_engines) > 0:
                engine = random.choice(rollout_engines)
                engine_version = ray.get(engine.get_weight_version.remote())
                if str(engine_version) != str(self.weight_updater.weight_version):
                    raise RuntimeError(
                        f"Weight version mismatch! Engine: {engine_version}, Updater: {self.weight_updater.weight_version}"
                    )

            if getattr(self.args, "keep_old_actor", False):
                if self.args.update_weights_interval == 1:
                    logger.info(
                        "updating model queue: rollout_actor -> old_actor, actor -> rollout_actor")
                    # Queue-style update: rollout_actor params -> old_actor, actor params -> rollout_actor
                    # First copy rollout_actor to old_actor
                    self.weights_backuper.copy(
                        src_tag="rollout_actor", dst_tag="old_actor")
                    # Then copy current actor to rollout_actor
                    self.weights_backuper.backup("rollout_actor")
                else:
                    self.weights_backuper.backup("old_actor")

        if self.args.offload_train:
            destroy_process_groups()

    def load_other_checkpoint(self, model_tag: str, path: str) -> None:
        logger.info(
            "rank=%s start loading %s checkpoint from %s",
            dist.get_rank(),
            model_tag,
            path,
        )
        dist.barrier(group=get_gloo_group())
        old_args = self.args.load, self.args.no_load_optim, self.args.no_load_rng, self.args.finetune
        self.args.load = path
        self.args.no_load_optim = True
        self.args.no_load_rng = True
        self.args.finetune = True

        old_ckpt_step = None
        if model_tag == "ref" and self.args.ref_ckpt_step is not None:
            old_ckpt_step = self.args.ckpt_step
            self.args.ckpt_step = self.args.ref_ckpt_step
        elif model_tag == "teacher" and self.args.opd_teacher_ckpt_step is not None:
            old_ckpt_step = self.args.ckpt_step
            self.args.ckpt_step = self.args.opd_teacher_ckpt_step

        _, _ = load_checkpoint(
            self.model,
            None,
            None,
            checkpointing_context={},
            skip_load_to_model_and_opt=False,
        )
        dist.barrier(group=get_gloo_group())
        logger.info(
            "rank=%s loaded %s checkpoint",
            dist.get_rank(),
            model_tag,
        )
        self.args.load, self.args.no_load_optim, self.args.no_load_rng, self.args.finetune = old_args

        if old_ckpt_step is not None:
            self.args.ckpt_step = old_ckpt_step

        logger.info(
            "rank=%s backing up %s checkpoint weights",
            dist.get_rank(),
            model_tag,
        )
        self.weights_backuper.backup(model_tag)
        dist.barrier(group=get_gloo_group())
        logger.info(
            "rank=%s finished backing up %s checkpoint",
            dist.get_rank(),
            model_tag,
        )
        self._active_model_tag = model_tag

    def connect_actor_critic(
        self,
        actor_handle: ActorHandle | None = None,
        master_address: str | None = None,
        master_port: int | None = None,
    ) -> None:
        if self.role == "actor":
            master_address = ray.util.get_node_ip_address()
            with socket.socket() as sock:
                sock.bind(("", 0))
                master_port = sock.getsockname()[1]
            actor_handle.connect_actor_critic.remote(
                master_address=master_address, master_port=master_port)

        group_name = "actor_critic"
        world_size = 2
        self._actor_critic_groups = init_process_group(
            backend="nccl",
            init_method=f"tcp://{master_address}:{master_port}",
            world_size=world_size,
            rank=0 if self.role == "actor" else 1,
            group_name=group_name,
        )
