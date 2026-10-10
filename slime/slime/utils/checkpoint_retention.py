import json
import logging
import re
import shutil
from pathlib import Path


logger = logging.getLogger(__name__)


_ITER_DIR_RE = re.compile(r"iter_(\d+)$")


def _iteration_dir_name(iteration: int) -> str:
    return f"iter_{iteration:07d}"


def _read_latest_iteration(save_dir: Path) -> int | None:
    latest_path = save_dir / "latest_checkpointed_iteration.txt"
    try:
        text = latest_path.read_text().strip()
    except OSError:
        return None

    try:
        return int(text)
    except ValueError:
        logger.warning(
            "Cannot parse latest checkpoint iteration from %s: %r", latest_path, text)
        return None


def _iter_checkpoint_dirs(save_dir: Path) -> list[tuple[int, Path]]:
    checkpoint_dirs: list[tuple[int, Path]] = []
    for path in save_dir.glob("iter_*"):
        if not path.is_dir():
            continue
        match = _ITER_DIR_RE.fullmatch(path.name)
        if match is None:
            continue
        checkpoint_dirs.append((int(match.group(1)), path))
    return checkpoint_dirs


def _write_best_metadata(
    save_dir: Path,
    *,
    best_rollout_id: int | None,
    best_metric: float | None,
    metric_name: str,
) -> None:
    if best_rollout_id is None or best_metric is None:
        return

    metadata = {
        "best_rollout_id": best_rollout_id,
        "best_iteration": best_rollout_id,
        "best_metric": best_metric,
        "metric_name": metric_name,
        "checkpoint_dir": _iteration_dir_name(best_rollout_id),
    }
    metadata_path = save_dir / "best_checkpoint.json"
    tmp_path = save_dir / "best_checkpoint.json.tmp"
    try:
        tmp_path.write_text(json.dumps(
            metadata, indent=2, sort_keys=True) + "\n")
        tmp_path.replace(metadata_path)
    except OSError:
        logger.exception(
            "Failed to write best checkpoint metadata to %s", metadata_path)


def _window_metadata_path(save_dir: Path) -> Path:
    return save_dir / "best_checkpoint_windows.json"


def _serialize_window_best(
    window_best: dict[tuple[int, int], tuple[float | None, int | None]],
    metric_name: str,
) -> dict:
    windows = []
    for (window_start, window_end), (best_metric, best_rollout_id) in sorted(window_best.items()):
        if best_metric is None or best_rollout_id is None:
            continue
        windows.append(
            {
                "window_start": window_start,
                "window_end": window_end,
                "best_rollout_id": best_rollout_id,
                "best_iteration": best_rollout_id,
                "best_metric": best_metric,
                "metric_name": metric_name,
                "checkpoint_dir": _iteration_dir_name(best_rollout_id),
            }
        )
    return {"metric_name": metric_name, "windows": windows}


def write_window_best_metadata(
    save_dirs: list[str | None],
    window_best: dict[tuple[int, int], tuple[float | None, int | None]],
    metric_name: str,
) -> None:
    for raw_save_dir in save_dirs:
        try:
            if raw_save_dir is None:
                continue
            save_dir = Path(raw_save_dir)
            if not save_dir.exists():
                continue
            metadata_path = _window_metadata_path(save_dir)
            tmp_path = save_dir / "best_checkpoint_windows.json.tmp"
            tmp_path.write_text(
                json.dumps(_serialize_window_best(
                    window_best, metric_name), indent=2, sort_keys=True) + "\n"
            )
            tmp_path.replace(metadata_path)
        except OSError:
            logger.exception(
                "Failed to write window best checkpoint metadata to %s", raw_save_dir)


def load_best_checkpoint_metadata(save_dir: str | None, metric_name: str) -> tuple[float | None, int | None]:
    if save_dir is None:
        return None, None

    metadata_path = Path(save_dir) / "best_checkpoint.json"
    try:
        metadata = json.loads(metadata_path.read_text())
    except OSError:
        return None, None
    except json.JSONDecodeError:
        logger.warning(
            "Cannot parse best checkpoint metadata from %s", metadata_path)
        return None, None

    if metadata.get("metric_name") != metric_name:
        return None, None

    try:
        best_metric = float(metadata["best_metric"])
        best_rollout_id = int(metadata["best_rollout_id"])
    except (KeyError, TypeError, ValueError):
        logger.warning("Invalid best checkpoint metadata in %s: %r",
                       metadata_path, metadata)
        return None, None

    return best_metric, best_rollout_id


def load_window_best_checkpoint_metadata(
    save_dir: str | None,
    metric_name: str,
) -> dict[tuple[int, int], tuple[float | None, int | None]]:
    if save_dir is None:
        return {}

    metadata_path = _window_metadata_path(Path(save_dir))
    try:
        metadata = json.loads(metadata_path.read_text())
    except OSError:
        return {}
    except json.JSONDecodeError:
        logger.warning(
            "Cannot parse window best checkpoint metadata from %s", metadata_path)
        return {}

    if metadata.get("metric_name") != metric_name:
        return {}

    window_best: dict[tuple[int, int], tuple[float | None, int | None]] = {}
    windows = metadata.get("windows") or []
    if not isinstance(windows, list):
        logger.warning(
            "Invalid window best checkpoint metadata in %s: %r", metadata_path, metadata)
        return {}

    for item in windows:
        try:
            window_start = int(item["window_start"])
            window_end = int(item["window_end"])
            best_metric = float(item["best_metric"])
            best_rollout_id = int(item["best_rollout_id"])
        except (KeyError, TypeError, ValueError):
            logger.warning(
                "Invalid window best checkpoint entry in %s: %r", metadata_path, item)
            continue
        window_best[(window_start, window_end)] = (
            best_metric, best_rollout_id)
    return window_best


def _read_existing_best_iteration(save_dir: Path, metric_name: str) -> int | None:
    _, best_rollout_id = load_best_checkpoint_metadata(
        str(save_dir), metric_name)
    return best_rollout_id


def _read_existing_window_best_iterations(save_dir: Path, metric_name: str) -> set[int]:
    window_best = load_window_best_checkpoint_metadata(
        str(save_dir), metric_name)
    return {
        best_rollout_id
        for _best_metric, best_rollout_id in window_best.values()
        if best_rollout_id is not None
    }


def retain_latest_and_best_checkpoints(
    save_dirs: list[str | None],
    *,
    latest_rollout_id: int,
    best_rollout_id: int | None,
    best_metric: float | None,
    metric_name: str,
    window_best: dict[tuple[int, int],
                      tuple[float | None, int | None]] | None = None,
    delete_old_checkpoints: bool = True,
) -> None:
    """Keep latest plus global/window-best Megatron iteration dirs in each save dir.

    When ``delete_old_checkpoints`` is false, this only updates latest/best
    metadata and keeps all checkpoint directories. It only removes directories
    named like ``iter_0000123`` when deletion is enabled.
    """

    for raw_save_dir in save_dirs:
        try:
            if raw_save_dir is None:
                continue

            save_dir = Path(raw_save_dir)
            if not save_dir.exists():
                continue

            latest_iteration = _read_latest_iteration(save_dir)
            if latest_iteration is None:
                latest_iteration = latest_rollout_id

            keep_iterations = {latest_iteration, latest_rollout_id}
            if best_rollout_id is not None:
                keep_iterations.add(best_rollout_id)
            else:
                existing_best_iteration = _read_existing_best_iteration(
                    save_dir, metric_name)
                if existing_best_iteration is not None:
                    keep_iterations.add(existing_best_iteration)

            if window_best:
                keep_iterations.update(
                    best_rollout_id
                    for _best_metric, best_rollout_id in window_best.values()
                    if best_rollout_id is not None
                )
            elif window_best is None:
                keep_iterations.update(
                    _read_existing_window_best_iterations(save_dir, metric_name))

            _write_best_metadata(
                save_dir,
                best_rollout_id=best_rollout_id,
                best_metric=best_metric,
                metric_name=metric_name,
            )
            if window_best is not None:
                write_window_best_metadata(
                    [raw_save_dir], window_best, metric_name)

            if not delete_old_checkpoints:
                continue

            for iteration, checkpoint_dir in _iter_checkpoint_dirs(save_dir):
                if iteration in keep_iterations:
                    continue
                try:
                    shutil.rmtree(checkpoint_dir)
                    logger.info(
                        "Removed old checkpoint directory %s", checkpoint_dir)
                except OSError:
                    logger.exception(
                        "Failed to remove old checkpoint directory %s", checkpoint_dir)
        except Exception:
            logger.exception(
                "Checkpoint retention failed for %s; continuing training.", raw_save_dir)
