#!/bin/bash

# Reuse the 8B SPPO recipe with Qwen3-4B-Base, as in the original script.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
TPPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
SLIME_ROOT="${TPPO_ROOT}/slime"
MEGATRON_LM_PATH="${MEGATRON_LM_PATH:-/path/to/Megatron-LM}"
export TPPO_ROOT SLIME_ROOT MEGATRON_LM_PATH
export PYTHONPATH="${SLIME_ROOT}:${MEGATRON_LM_PATH}:${PYTHONPATH:-}"
export METHOD="${METHOD:-SPPO}"
export EXP_NAME="${EXP_NAME:-4B-Base-SPPO-Polaris-n8}"
export ADVANTAGE_ESTIMATOR="${ADVANTAGE_ESTIMATOR:-sppo}"
export N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-8}"
export HF_MODEL_PATH="${HF_MODEL_PATH:-/path/to/models/Qwen3-4B-Base/}"
export MEGATRON_MODEL_PATH="${MEGATRON_MODEL_PATH:-/path/to/models/Qwen3-4B-Base-megatron}"
export MODEL_CONFIG_PATH="${MODEL_CONFIG_PATH:-${SLIME_ROOT}/scripts/models/qwen3-4B.sh}"

exec bash "${SCRIPT_DIR}/../8b_polaris/sppo.sh" "$@"
