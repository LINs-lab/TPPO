#!/bin/bash

# SPPO launch: keep every runtime/model/data setting from ppo.sh and
# only switch the estimator plus the number of responses per prompt.
export METHOD="${METHOD:-SPPO}"
export EXP_NAME="${EXP_NAME:-8B-Base-SPPO-Polaris-n8}"
export ADVANTAGE_ESTIMATOR="${ADVANTAGE_ESTIMATOR:-sppo}"
export N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-8}"
export NUM_ROLLOUT="${NUM_ROLLOUT:-520}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-10}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
TPPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
SLIME_ROOT="${TPPO_ROOT}/slime"
MEGATRON_LM_PATH="${MEGATRON_LM_PATH:-/path/to/Megatron-LM}"
export TPPO_ROOT SLIME_ROOT MEGATRON_LM_PATH
export PYTHONPATH="${SLIME_ROOT}:${MEGATRON_LM_PATH}:${PYTHONPATH:-}"
exec bash "${SCRIPT_DIR}/ppo.sh" "$@"
