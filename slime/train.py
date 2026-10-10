import logging
import os

import ray

from slime.ray.placement_group import create_placement_groups, create_rollout_manager, create_training_models
from slime.utils.arguments import parse_args
try:
    from slime.utils.checkpoint_retention import (
        load_best_checkpoint_metadata,
        load_window_best_checkpoint_metadata,
        retain_latest_and_best_checkpoints,
        write_window_best_metadata,
    )
except ImportError:
    # Analysis export never saves/retains checkpoints.  Keep this mode usable
    # with an older installed Slime package that predates window-best metadata;
    # normal training still fails loudly instead of silently disabling it.
    if os.environ.get("ANALYSIS_EXPORT_ONLY", "0") != "1":
        raise

    def load_best_checkpoint_metadata(save_dir, metric_name):
        return None, None

    def load_window_best_checkpoint_metadata(save_dir, metric_name):
        return {}

    def retain_latest_and_best_checkpoints(*args, **kwargs):
        return None

    def write_window_best_metadata(*args, **kwargs):
        return None
from slime.utils.logging_utils import configure_logger, finish_tracking, init_tracking, update_tracking_open_metrics
from slime.utils.misc import should_run_periodic_action
from slime.utils.add_my_custom_arguments import add_my_custom_arguments


logger = logging.getLogger(__name__)


def _build_save_best_windows(window_ends):
    if not window_ends:
        return []

    windows = []
    window_start = 0
    for raw_window_end in window_ends:
        window_end = int(raw_window_end)
        if window_end <= window_start:
            raise ValueError(
                "--save-best-window-ends must be strictly increasing positive rollout counts; "
                f"got end={window_end} after start={window_start}."
            )
        windows.append((window_start, window_end))
        window_start = window_end
    return windows


def _save_best_window_for_rollout(rollout_id, windows):
    for window_start, window_end in windows:
        if window_start <= rollout_id < window_end:
            return (window_start, window_end)
    return None


def train(args):
    configure_logger()
    analysis_export_only = bool(getattr(args, "analysis_export_only", False))
    save_latest_best_ckpts = getattr(args, "save_latest_best_ckpts", False)
    save_all_ckpts = getattr(args, "save_all_ckpts", False)
    save_only_best = getattr(args, "save_only_best", False)
    save_best_metric = getattr(args, "save_best_metric", "eval/overall")
    save_interval_offset = getattr(args, "save_interval_offset", 0)
    best_save_dir = getattr(args, "best_save", None)
    critic_best_save_dir = getattr(args, "critic_best_save", None)
    if save_latest_best_ckpts:
        if best_save_dir is None and args.save is not None:
            best_save_dir = os.path.join(str(args.save).rstrip("/"), "best")
        if (
            critic_best_save_dir is None
            and getattr(args, "critic_save", None) is not None
        ):
            critic_best_save_dir = os.path.join(
                str(args.critic_save).rstrip("/"), "best")
    separate_best_checkpoints = best_save_dir is not None
    save_best_windows = _build_save_best_windows(
        getattr(args, "save_best_window_ends", None))
    if analysis_export_only:
        logger.info(
            "Running in analysis export only mode: rollout/forward/export without optimizer updates.")
    if save_latest_best_ckpts and not analysis_export_only:
        assert args.save_interval is not None, "'--save-interval' is required when save_latest_best_ckpts is set."
        assert args.eval_interval is not None, "'--eval-interval' is required when save_latest_best_ckpts is set."
        if args.save_interval != args.eval_interval:
            logger.warning(
                "--save-interval (%s) and --eval-interval (%s) differ. "
                "latest will be updated at every save point; best/window-best "
                "will only be updated when a save point and an eval point coincide.",
                args.save_interval,
                args.eval_interval,
            )
        if save_best_windows:
            logger.info(
                "Enabled window best checkpoint retention for windows: %s", save_best_windows)
        if save_only_best:
            logger.info(
                "Regular iter_* checkpoint saves are disabled; only the current best checkpoint will be saved."
            )

    # allocate the GPUs
    pgs = create_placement_groups(args)
    init_tracking(args)

    # create the rollout manager, with sglang engines inside.
    # need to initialize rollout manager first to calculate num_rollout
    rollout_manager, num_rollout_per_epoch = create_rollout_manager(
        args, pgs["rollout"])

    # Update primary W&B with SGLang metrics endpoint now that servers are up.
    router_addr = ray.get(rollout_manager.get_metrics_router_addr.remote())
    update_tracking_open_metrics(args, router_addr)

    # create the actor and critic models
    actor_model, critic_model = create_training_models(
        args, pgs, rollout_manager)

    if args.offload_rollout:
        ray.get(rollout_manager.onload_weights.remote())

    # always update weight first so that sglang has the loaded weights from training.
    if not args.critic_train_only:
        actor_model.update_weights()

        if args.check_weight_update_equal:
            ray.get(rollout_manager.check_weights.remote(action="compare"))

    if args.offload_rollout:
        ray.get(rollout_manager.onload_kv.remote())

    # special case for eval-only
    if args.num_rollout == 0 and args.eval_interval is not None:
        ray.get(rollout_manager.eval.remote(rollout_id=0))

    def offload_train(rollout_id):
        if args.offload_train:
            if args.use_critic:
                critic_model.offload()
                if rollout_id >= args.num_critic_only_steps and not args.critic_train_only:
                    actor_model.offload()
            else:
                actor_model.offload()
        else:
            if args.critic_train_only:
                critic_model.clear_memory()
            else:
                actor_model.clear_memory()

    best_metric = None
    best_rollout_id = None
    window_best: dict[tuple[int, int], tuple[float | None, int | None]] = {
        window: (None, None) for window in save_best_windows
    }
    regular_best_metric = None
    regular_best_rollout_id = None
    regular_window_best: dict[tuple[int, int], tuple[float | None, int | None]] = {
        window: (None, None) for window in save_best_windows
    }
    # A fresh best-only run must rank its own evaluations, even when the
    # output directory still contains metadata from an earlier run. Resume
    # runs keep the existing best so they can continue the same comparison.
    if save_latest_best_ckpts and (not save_only_best or args.start_rollout_id > 0):
        metadata_dir = best_save_dir if separate_best_checkpoints else args.save
        best_metric, best_rollout_id = load_best_checkpoint_metadata(
            metadata_dir, save_best_metric)
        if best_rollout_id is not None:
            logger.info(
                "Loaded existing best checkpoint at rollout %s: %s=%s",
                best_rollout_id,
                save_best_metric,
                best_metric,
            )
        loaded_window_best = load_window_best_checkpoint_metadata(
            metadata_dir, save_best_metric)
        for window in window_best:
            if window in loaded_window_best:
                window_best[window] = loaded_window_best[window]
        for (window_start, window_end), (metric, rollout_id) in window_best.items():
            if rollout_id is not None:
                logger.info(
                    "Loaded existing window best checkpoint for [%s, %s): rollout %s, %s=%s",
                    window_start,
                    window_end,
                    rollout_id,
                    save_best_metric,
                    metric,
                )
        if separate_best_checkpoints:
            regular_best_metric, regular_best_rollout_id = load_best_checkpoint_metadata(
                args.save, save_best_metric)
            loaded_regular_window_best = load_window_best_checkpoint_metadata(
                args.save, save_best_metric)
            for window in regular_window_best:
                if window in loaded_regular_window_best:
                    regular_window_best[window] = loaded_regular_window_best[window]

    def save(
        rollout_id,
        actor_save_dir=None,
        critic_save_dir=None,
        force_sync=None,
    ):
        saved_dirs = []
        actor_saved = False
        if force_sync is None:
            force_sync = rollout_id == args.num_rollout - 1 or save_latest_best_ckpts
        if (not args.use_critic) or (rollout_id >= args.num_critic_only_steps and not args.critic_train_only):
            actor_model.save_model(
                rollout_id,
                force_sync=force_sync,
                save_dir=actor_save_dir,
            )
            saved_dirs.append(actor_save_dir or args.save)
            actor_saved = True
        if args.use_critic:
            critic_model.save_model(
                rollout_id,
                force_sync=force_sync,
                save_dir=critic_save_dir,
            )
            saved_dirs.append(critic_save_dir or args.critic_save)
        if args.rollout_global_dataset:
            ray.get(rollout_manager.save.remote(rollout_id))
        return saved_dirs, actor_saved

    def retain_checkpoints(
        rollout_id,
        saved_dirs,
        best_rollout_id=None,
        best_metric=None,
        window_best=None,
        delete_old_checkpoints=None,
    ):
        if not save_latest_best_ckpts or not saved_dirs:
            return
        retain_latest_and_best_checkpoints(
            saved_dirs,
            latest_rollout_id=rollout_id,
            best_rollout_id=best_rollout_id,
            best_metric=best_metric,
            metric_name=save_best_metric,
            window_best=window_best,
            delete_old_checkpoints=(
                not save_all_ckpts
                if delete_old_checkpoints is None
                else delete_old_checkpoints
            ),
        )

    def update_best_checkpoint(rollout_id, eval_metrics, saved_dirs, actor_saved):
        nonlocal best_metric, best_rollout_id, regular_best_metric, regular_best_rollout_id

        if not save_latest_best_ckpts:
            return
        # With --save-only-best there is no regular save before evaluation.
        # The evaluated actor can still be saved if it becomes the new best.
        actor_ready_for_best_save = (
            not args.use_critic or rollout_id >= args.num_critic_only_steps
        )
        if not (
            actor_saved
            or args.critic_train_only
            or (save_only_best and actor_ready_for_best_save)
        ):
            return
        if not eval_metrics:
            logger.warning(
                "Skip best checkpoint update at rollout %s because eval returned no metrics.", rollout_id)
            return

        metric = eval_metrics.get(save_best_metric)
        if metric is None:
            logger.warning(
                "Skip best checkpoint update at rollout %s because metric %s is missing from eval metrics.",
                rollout_id,
                save_best_metric,
            )
            return

        try:
            metric = float(metric)
        except (TypeError, ValueError):
            logger.warning(
                "Skip best checkpoint update at rollout %s because metric %s is not numeric: %r",
                rollout_id,
                save_best_metric,
                metric,
            )
            return

        window = _save_best_window_for_rollout(rollout_id, save_best_windows)

        if separate_best_checkpoints:
            best_updated = False
            if best_metric is None or metric > best_metric:
                best_metric = metric
                best_rollout_id = rollout_id
                best_updated = True
                logger.info(
                    "New best checkpoint at rollout %s: %s=%s",
                    best_rollout_id,
                    save_best_metric,
                    best_metric,
                )
            if window is not None:
                window_metric, _window_rollout_id = window_best.get(
                    window, (None, None))
                if window_metric is None or metric > window_metric:
                    window_best[window] = (metric, rollout_id)
                    best_updated = True
                    logger.info(
                        "New window best checkpoint for [%s, %s) at rollout %s: %s=%s",
                        window[0],
                        window[1],
                        rollout_id,
                        save_best_metric,
                        metric,
                    )
            if best_updated:
                was_offloaded = bool(args.offload_train)
                rollout_offloaded_for_best_save = False
                if was_offloaded:
                    # Evaluation runs with the rollout engines resident.  In a
                    # colocated PPO job there is not enough GPU memory to wake
                    # both training models while the rollout weights, KV cache,
                    # and CUDA graphs are still resident.  The ordinary train
                    # path already offloads rollout before waking training;
                    # mirror that ordering for the separate best-checkpoint
                    # save path as well.
                    if args.offload_rollout:
                        logger.info(
                            "Offloading rollout engines before saving best checkpoint at rollout %s.",
                            rollout_id,
                        )
                        ray.get(rollout_manager.offload.remote())
                        rollout_offloaded_for_best_save = True
                    if args.use_critic:
                        critic_model.onload()
                    if not args.critic_train_only:
                        actor_model.onload()
                try:
                    best_saved_dirs, best_actor_saved = save(
                        rollout_id,
                        actor_save_dir=best_save_dir,
                        critic_save_dir=critic_best_save_dir,
                        force_sync=True,
                    )
                    retain_checkpoints(
                        rollout_id,
                        best_saved_dirs,
                        best_rollout_id=best_rollout_id,
                        best_metric=best_metric,
                        window_best=window_best,
                        delete_old_checkpoints=True,
                    )
                finally:
                    if was_offloaded:
                        offload_train(rollout_id)
                    if rollout_offloaded_for_best_save:
                        logger.info(
                            "Restoring rollout engines after saving best checkpoint at rollout %s.",
                            rollout_id,
                        )
                        ray.get(rollout_manager.onload_weights.remote())
                        # Without a CPU weight backup, resuming memory alone
                        # does not restore the actor weights used for generation.
                        if not args.critic_train_only:
                            actor_model.update_weights()
                        ray.get(rollout_manager.onload_kv.remote())

            if not saved_dirs:
                return

            regular_best_updated = False
            if regular_best_metric is None or metric > regular_best_metric:
                regular_best_metric = metric
                regular_best_rollout_id = rollout_id
                regular_best_updated = True
            if window is not None:
                regular_window_metric, _regular_window_rollout_id = regular_window_best.get(
                    window, (None, None))
                if regular_window_metric is None or metric > regular_window_metric:
                    regular_window_best[window] = (metric, rollout_id)
                    regular_best_updated = True
            if regular_best_updated:
                retain_checkpoints(
                    rollout_id,
                    saved_dirs,
                    best_rollout_id=regular_best_rollout_id,
                    best_metric=regular_best_metric,
                    window_best=regular_window_best,
                )
            return

        if not saved_dirs:
            return

        best_updated = False
        if best_metric is None or metric > best_metric:
            best_metric = metric
            best_rollout_id = rollout_id
            best_updated = True
            logger.info(
                "New best checkpoint at rollout %s: %s=%s",
                best_rollout_id,
                save_best_metric,
                best_metric,
            )
        if window is not None:
            window_metric, _window_rollout_id = window_best.get(
                window, (None, None))
            if window_metric is None or metric > window_metric:
                window_best[window] = (metric, rollout_id)
                best_updated = True
                logger.info(
                    "New window best checkpoint for [%s, %s) at rollout %s: %s=%s",
                    window[0],
                    window[1],
                    rollout_id,
                    save_best_metric,
                    metric,
                )
        if best_updated:
            retain_checkpoints(
                rollout_id,
                saved_dirs,
                best_rollout_id=best_rollout_id,
                best_metric=best_metric,
                window_best=window_best,
            )
    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        saved_dirs = []
        actor_saved = False
        if args.eval_interval is not None and rollout_id == 0 and not args.skip_eval_before_train:
            ray.get(rollout_manager.eval.remote(rollout_id))
        rollout_data_ref = ray.get(rollout_manager.generate.remote(rollout_id))

        if args.offload_rollout:
            ray.get(rollout_manager.offload.remote())

        if args.use_critic:
            actor_trains_this_step = (
                rollout_id >= args.num_critic_only_steps and not args.critic_train_only
            )
            if analysis_export_only:
                critic_export_handle = critic_model.async_analysis_export(
                    rollout_id, rollout_data_ref)
                if not args.critic_train_only:
                    ray.get(actor_model.async_analysis_export(
                        rollout_id, rollout_data_ref))
                ray.get(critic_export_handle)
            else:
                critic_train_handle = critic_model.async_train(
                    rollout_id, rollout_data_ref)
                if actor_trains_this_step:
                    ray.get(actor_model.async_train(
                        rollout_id, rollout_data_ref))
                ray.get(critic_train_handle)
        else:
            if analysis_export_only:
                ray.get(actor_model.async_analysis_export(
                    rollout_id, rollout_data_ref))
            else:
                ray.get(actor_model.async_train(rollout_id, rollout_data_ref))

        if analysis_export_only:
            continue

        if not save_only_best and should_run_periodic_action(
            rollout_id,
            args.save_interval,
            num_rollout_per_epoch,
            args.num_rollout,
            offset=save_interval_offset,
        ):
            saved_dirs, actor_saved = save(rollout_id)
            retain_checkpoints(
                rollout_id,
                saved_dirs,
                best_rollout_id=(
                    regular_best_rollout_id
                    if separate_best_checkpoints
                    else best_rollout_id
                ),
                best_metric=(
                    regular_best_metric
                    if separate_best_checkpoints
                    else best_metric
                ),
                window_best=(
                    regular_window_best
                    if separate_best_checkpoints
                    else window_best
                ),
            )

        offload_train(rollout_id)
        if args.offload_rollout:
            ray.get(rollout_manager.onload_weights.remote())
        if not args.critic_train_only:
            actor_model.update_weights()
        if args.offload_rollout:
            ray.get(rollout_manager.onload_kv.remote())

        skip_warmup_eval = (
            getattr(args, "skip_eval_during_critic_warmup", False)
            and rollout_id < args.num_critic_only_steps
        )
        if (
            not skip_warmup_eval
            and should_run_periodic_action(rollout_id, args.eval_interval, num_rollout_per_epoch)
        ):
            eval_metrics = ray.get(rollout_manager.eval.remote(rollout_id))
            update_best_checkpoint(
                rollout_id, eval_metrics, saved_dirs, actor_saved)

    ray.get(rollout_manager.dispose.remote())
    finish_tracking(args)


if __name__ == "__main__":
    args = parse_args(add_custom_arguments=add_my_custom_arguments)
    train(args)
