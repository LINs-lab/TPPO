#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TPPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
SLIME_ROOT="${TPPO_ROOT}/slime"
MEGATRON_LM_PATH="${MEGATRON_LM_PATH:-/path/to/Megatron-LM}"
export TPPO_ROOT SLIME_ROOT MEGATRON_LM_PATH
export PYTHONPATH="${SLIME_ROOT}:${MEGATRON_LM_PATH}:${PYTHONPATH:-}"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    cat <<'EOF'
Example:
  ACTOR_CHECKPOINT=/path/to/actor/iter_000119 \
  CRITIC_CHECKPOINT=/path/to/critic/iter_000119 \
  ORIGIN_HF=/path/to/models/Qwen3-8B-Base \
  TRAIN_TP=4 ACTOR_TP=2 \
  bash analysis/manual_mc/run_manual_mc_example.sh

Required: ACTOR_CHECKPOINT, CRITIC_CHECKPOINT
Optional: MC_MODE (baseline|tppo|tppo_separate_v0_head_logit_bce),
          MC_DATASET, MC_OUTPUT_DIR, MC_LIMIT, MC_ROLLOUTS, MC_SAMPLES,
          TRAIN_TP, ACTOR_TP, ORIGIN_HF, CRITIC_HEAD
MC_LIMIT defaults to 2 for a short test. Set MC_LIMIT=0 for the full dataset.
EOF
    exit 0
fi

: "${ACTOR_CHECKPOINT:?Set ACTOR_CHECKPOINT to an actor checkpoint directory.}"
: "${CRITIC_CHECKPOINT:?Set CRITIC_CHECKPOINT to a critic checkpoint directory.}"

MC_MODE="${MC_MODE:-tppo}"
MC_DATASET="${MC_DATASET:-${TPPO_ROOT}/data/eval/aime_2024.jsonl}"
MC_OUTPUT_DIR="${MC_OUTPUT_DIR:-${TMPDIR:-/tmp}/tppo_manual_mc_example}"
MC_LIMIT="${MC_LIMIT:-2}"
MC_ROLLOUTS="${MC_ROLLOUTS:-8}"
MC_SAMPLES="${MC_SAMPLES:-128}"
TRAIN_TP="${TRAIN_TP:-4}"
ACTOR_TP="${ACTOR_TP:-2}"
PAIR_NAME="${PAIR_NAME:-manual_mc_example}"

for checkpoint in "${ACTOR_CHECKPOINT}" "${CRITIC_CHECKPOINT}"; do
    if [[ ! -d "${checkpoint}" ]]; then
        echo "Checkpoint directory does not exist: ${checkpoint}" >&2
        exit 1
    fi
done
if [[ ! -f "${MC_DATASET}" ]]; then
    echo "Dataset does not exist: ${MC_DATASET}" >&2
    exit 1
fi

if [[ ! -f "${ACTOR_CHECKPOINT}/config.json" || ! -f "${CRITIC_CHECKPOINT}/config.json" ]]; then
    : "${ORIGIN_HF:?Set ORIGIN_HF when converting Megatron checkpoints to Hugging Face format.}"
    if [[ ! -d "${ORIGIN_HF}" ]]; then
        echo "Origin Hugging Face model directory does not exist: ${ORIGIN_HF}" >&2
        exit 1
    fi
fi

mkdir -p "${MC_OUTPUT_DIR}"
PAIR_FILE="$(mktemp "${TMPDIR:-/tmp}/tppo-manual-mc-pair.XXXXXX.jsonl")"
trap 'rm -f "${PAIR_FILE}"' EXIT
MC_PAIR_NAME="${PAIR_NAME}" MC_MODE="${MC_MODE}" \
MC_ACTOR_CHECKPOINT="${ACTOR_CHECKPOINT}" MC_CRITIC_CHECKPOINT="${CRITIC_CHECKPOINT}" \
MC_CRITIC_HEAD="${CRITIC_HEAD:-}" MC_PAIR_FILE="${PAIR_FILE}" \
python3 - <<'PY'
import json
import os

pair = {
    "name": os.environ["MC_PAIR_NAME"],
    "mode": os.environ["MC_MODE"],
    "actor": os.environ["MC_ACTOR_CHECKPOINT"],
    "critic": os.environ["MC_CRITIC_CHECKPOINT"],
}
if os.environ["MC_CRITIC_HEAD"]:
    pair["head"] = os.environ["MC_CRITIC_HEAD"]
with open(os.environ["MC_PAIR_FILE"], "w", encoding="utf-8") as handle:
    handle.write(json.dumps(pair) + "\n")
PY

MC_ARGS=(
    --pairs "${PAIR_FILE}"
    --pair-names "${PAIR_NAME}"
    --dataset "${MC_DATASET}"
    --out "${MC_OUTPUT_DIR}"
    --limit "${MC_LIMIT}"
    --rollouts "${MC_ROLLOUTS}"
    --mc-samples "${MC_SAMPLES}"
    --train-tp "${TRAIN_TP}"
    --actor-tp "${ACTOR_TP}"
)
if [[ -n "${ORIGIN_HF:-}" ]]; then
    MC_ARGS+=(--origin-hf "${ORIGIN_HF}")
fi
if [[ -n "${CRITIC_HEAD:-}" ]]; then
    MC_ARGS+=(--head "${CRITIC_HEAD}")
fi

echo "Running ${MC_MODE} manual MC on ${MC_DATASET} (limit=${MC_LIMIT}, rollouts=${MC_ROLLOUTS}, MC samples=${MC_SAMPLES})"
echo "Results: ${MC_OUTPUT_DIR}/${PAIR_NAME}"
python3 "${SCRIPT_DIR}/manual_mc.py" "${MC_ARGS[@]}"
