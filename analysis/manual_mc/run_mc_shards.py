#!/usr/bin/env python3
"""Run manual MC on independent TP=1 replicas and merge the results."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
MANUAL_MC = SCRIPT_DIR / "manual_mc.py"
REPO_ROOT = SCRIPT_DIR.parents[1] / "slime"


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [
            json.loads(line)
            for line in f
            if line.strip() and not line.lstrip().startswith("#")
        ]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--pairs", type=Path, required=True)
    p.add_argument("--pair-names", nargs="+", required=True)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--rollouts", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=8192)
    p.add_argument("--mc-samples", type=int, default=128)
    p.add_argument("--mc-ratios", type=float, nargs="+", required=True)
    p.add_argument("--mc-reward-concurrency", type=int, default=128)
    p.add_argument("--sglang-concurrency", type=int, default=64)
    p.add_argument("--sglang-mem-fraction-static", type=float, default=0.85)
    p.add_argument("--base-port", type=int, default=30020)
    p.add_argument("--devices", required=True, help="Comma-separated GPU ids, e.g. 0,1,2,3,4,5,6,7")
    return p.parse_args()


def split_rows(rows: list[dict], count: int) -> list[tuple[int, list[dict]]]:
    result = []
    base, extra = divmod(len(rows), count)
    start = 0
    for shard_id in range(count):
        size = base + int(shard_id < extra)
        result.append((start, rows[start:start + size]))
        start += size
    return result


def wait_for_hf_model(path: Path, process: subprocess.Popen, timeout: float = 900.0) -> None:
    deadline = time.time() + timeout
    while True:
        code = process.poll()
        if code is not None:
            raise RuntimeError(f"bootstrap shard exited with code {code} before creating a complete HF model at {path}")
        complete = (path / "config.json").is_file()
        index = path / "model.safetensors.index.json"
        if index.is_file():
            try:
                weight_map = json.loads(index.read_text(encoding="utf-8")).get("weight_map", {})
                complete = complete and all((path / name).is_file() and (path / name).stat().st_size > 0
                                            for name in set(weight_map.values()))
            except (OSError, json.JSONDecodeError):
                complete = False
        else:
            complete = complete and (path / "model.safetensors").is_file()
        if complete:
            return
        if time.time() >= deadline:
            raise TimeoutError(f"timed out waiting for a complete converted HF model: {path}")
        time.sleep(2)


def run_pair(args: argparse.Namespace, pair: dict, rows: list[dict], devices: list[str], pair_index: int) -> None:
    if not rows:
        raise ValueError("cannot shard an empty dataset")
    shard_count = min(len(devices), len(rows))
    shards = split_rows(rows, shard_count)
    pair_name = str(pair["name"])
    args.out.mkdir(parents=True, exist_ok=True)
    tmp_root = Path(tempfile.mkdtemp(prefix=f".mc_shards_{pair_name}_", dir=args.out))
    processes: list[subprocess.Popen] = []
    handles = []
    try:
        original_pair_file = tmp_root / "pairs_original.jsonl"
        write_jsonl(original_pair_file, [pair])
        shard_pair_file = tmp_root / "pairs_shared_hf.jsonl"

        shard_outputs: list[Path] = []
        for shard_id, (offset, shard_rows) in enumerate(shards):
            shard_dataset = tmp_root / f"dataset_{shard_id:02d}.jsonl"
            write_jsonl(shard_dataset, shard_rows)
            shard_outputs.append(tmp_root / f"output_{shard_id:02d}")

        actor_source = Path(pair["actor"])
        actor_is_hf = (actor_source / "config.json").is_file()
        bootstrap_hf = shard_outputs[0] / pair_name / "hf" / "actor"
        if actor_is_hf:
            write_jsonl(shard_pair_file, [pair])
        else:
            write_jsonl(shard_pair_file, [pair])

        def launch(shard_id: int, pair_file: Path, keep_hf: bool = False) -> subprocess.Popen:
            offset, _ = shards[shard_id]
            log_path = args.out / "logs" / f"{pair_name}.shard_{shard_id:02d}.log"
            sglang_log_path = args.out / "logs" / f"{pair_name}.shard_{shard_id:02d}.sglang.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            cmd = [
                sys.executable,
                str(MANUAL_MC),
                "--pairs", str(pair_file),
                "--pair-names", pair_name,
                "--dataset", str(tmp_root / f"dataset_{shard_id:02d}.jsonl"),
                "--out", str(shard_outputs[shard_id]),
                "--limit", "0",
                "--prompt-offset", str(offset),
                "--rollouts", str(args.rollouts),
                "--max-new-tokens", str(args.max_new_tokens),
                "--mc-samples", str(args.mc_samples),
                "--mc-ratios", *[str(x) for x in args.mc_ratios],
                "--mc-reward-concurrency", str(args.mc_reward_concurrency),
                "--actor-tp", "1",
                "--sglang-port", str(args.base_port + pair_index * shard_count + shard_id),
                "--sglang-concurrency", str(args.sglang_concurrency),
                "--sglang-mem-fraction-static", str(args.sglang_mem_fraction_static),
                "--sglang-log-path", str(sglang_log_path),
                "--skip-critic",
            ]
            if not actor_is_hf and shard_id > 0:
                cmd.append("--no-convert")
            if keep_hf:
                cmd.append("--keep-generated-hf")
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = devices[shard_id]
            env["PYTHONUNBUFFERED"] = "1"
            # The manual script imports the exact reward implementation from
            # the sibling Slime checkout.  Make this explicit here as well as
            # in run_manual_mc.sh, so direct invocations of this shard runner
            # cannot silently lose the import path in child processes.
            env["PYTHONPATH"] = str(REPO_ROOT)
            handle = log_path.open("w", encoding="utf-8")
            handles.append(handle)
            process = subprocess.Popen(cmd, env=env, stdout=handle, stderr=subprocess.STDOUT, text=True)
            processes.append(process)
            return process

        bootstrap_process = launch(0, original_pair_file, keep_hf=not actor_is_hf)
        if not actor_is_hf:
            wait_for_hf_model(bootstrap_hf, bootstrap_process)
            shared_pair = dict(pair)
            shared_pair["actor"] = str(bootstrap_hf)
            write_jsonl(shard_pair_file, [shared_pair])

        for shard_id in range(1, shard_count):
            launch(shard_id, shard_pair_file)

        exit_codes = [process.wait() for process in processes]
        if any(code != 0 for code in exit_codes):
            raise RuntimeError(f"one or more MC shards failed for {pair_name}: {exit_codes}")

        merged_trace_rows: list[dict] = []
        merged_samples = []
        for shard_id, (offset, shard_rows) in enumerate(shards):
            shard_root = shard_outputs[shard_id] / pair_name
            trace_path = shard_root / "value_traces.jsonl"
            pt_path = shard_root / "runtime" / "debug_data" / "analysis" / "training_details.json" / "rollout_data" / "0.pt"
            if not trace_path.is_file() or not pt_path.is_file():
                raise FileNotFoundError(f"missing shard output for {pair_name}, shard {shard_id}")
            shard_trace_rows = read_jsonl(trace_path)
            expected = len(shard_rows) * args.rollouts
            if len(shard_trace_rows) != expected:
                raise ValueError(f"shard {shard_id} has {len(shard_trace_rows)} traces, expected {expected}")
            merged_trace_rows.extend(shard_trace_rows)
            try:
                import torch
                payload = torch.load(pt_path, map_location="cpu", weights_only=False)
            except TypeError:
                payload = torch.load(pt_path, map_location="cpu")
            merged_samples.extend(payload.get("samples", []))

        merged_trace_rows.sort(key=lambda row: int(row["sample_index"]))
        merged_samples.sort(key=lambda sample: int(sample["index"]))
        expected_indices = list(range(len(rows) * args.rollouts))
        actual_indices = [int(row["sample_index"]) for row in merged_trace_rows]
        if actual_indices != expected_indices:
            raise ValueError("merged sample_index values are not contiguous/global-aligned")
        if [int(sample["index"]) for sample in merged_samples] != expected_indices:
            raise ValueError("merged Slime sample indices are not contiguous/global-aligned")

        final_root = args.out / pair_name
        write_jsonl(final_root / "value_traces.jsonl", merged_trace_rows)
        final_pt = final_root / "runtime" / "debug_data" / "analysis" / "training_details.json" / "rollout_data" / "0.pt"
        final_pt.parent.mkdir(parents=True, exist_ok=True)
        import torch
        torch.save({"samples": merged_samples}, final_pt)
        print(f"{pair_name}: merged {len(merged_trace_rows)} rollouts from {shard_count} TP=1 shards -> {final_root}", flush=True)
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for handle in handles:
            handle.close()
        shutil.rmtree(tmp_root, ignore_errors=True)


def main() -> None:
    args = parse_args()
    devices = [item.strip() for item in args.devices.split(",") if item.strip()]
    if not devices:
        raise ValueError("--devices must contain at least one GPU id")
    rows = read_jsonl(args.dataset)
    if args.limit:
        rows = rows[:args.limit]
    pairs = {str(pair["name"]): pair for pair in read_jsonl(args.pairs)}
    for pair_index, pair_name in enumerate(args.pair_names):
        if pair_name not in pairs:
            raise KeyError(f"pair {pair_name!r} not found in {args.pairs}")
        run_pair(args, pairs[pair_name], rows, devices, pair_index)


if __name__ == "__main__":
    main()
