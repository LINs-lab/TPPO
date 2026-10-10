#!/bin/bash

# Compare SPPO n=1 with the local PPO: same Qwen3-4B, thinking disabled,
# prompt batch size 128, two steps per rollout, and critic warmup 50.
export METHOD="${METHOD:-SPPO}"
export EXP_NAME="${EXP_NAME:-4B-NoThink-SPPO-Polaris-n1-bs128-w50}"
export ADVANTAGE_ESTIMATOR="${ADVANTAGE_ESTIMATOR:-sppo}"
export N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-1}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
TPPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
SLIME_ROOT="${TPPO_ROOT}/slime"
MEGATRON_LM_PATH="${MEGATRON_LM_PATH:-/path/to/Megatron-LM}"
export TPPO_ROOT SLIME_ROOT MEGATRON_LM_PATH
export PYTHONPATH="${SLIME_ROOT}:${MEGATRON_LM_PATH}:${PYTHONPATH:-}"
exec bash "${SCRIPT_DIR}/ppo.sh" "$@"
