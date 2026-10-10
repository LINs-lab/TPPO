#!/bin/bash

set -euo pipefail
set -o pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
TPPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
SLIME_ROOT="${TPPO_ROOT}/slime"
MEGATRON_LM_PATH="${MEGATRON_LM_PATH:-/path/to/Megatron-LM}"
export TPPO_ROOT SLIME_ROOT MEGATRON_LM_PATH
export PYTHONPATH="${SLIME_ROOT}:${MEGATRON_LM_PATH}:${PYTHONPATH:-}"
PROJECT_ROOT="${SLIME_ROOT}"

VALUE_LOSS="${PPO_VALUE_LOSS:-bce}"
case "${VALUE_LOSS}" in
    bce|mse) ;;
    *)
        echo "PPO_VALUE_LOSS must be bce or mse; got: ${VALUE_LOSS}" >&2
        exit 2
        ;;
esac

HF_CHECKPOINT="${HF_CHECKPOINT:-/path/to/models/Qwen2.5-Math-1.5B}"
MEGATRON_CHECKPOINT="${MEGATRON_CHECKPOINT:-/path/to/models/Qwen2.5-Math-1.5B-megatron}"
TRAIN_DATA="${TRAIN_DATA:-${TPPO_ROOT}/data/countdown_3to4/train.jsonl}"
EVAL_DATA="${EVAL_DATA:-${TPPO_ROOT}/data/countdown_3to4/test.jsonl}"
OOD_EVAL_DATA="${OOD_EVAL_DATA:-${TPPO_ROOT}/data/countdown_5_ood/test.jsonl}"
ENABLE_EVAL="${ENABLE_EVAL:-1}"
ENABLE_OOD_EVAL="${ENABLE_OOD_EVAL:-1}"

required_files=(
    "${HF_CHECKPOINT}/config.json"
    "${MEGATRON_CHECKPOINT}/latest_checkpointed_iteration.txt"
    "${TRAIN_DATA}"
    "${EVAL_DATA}"
)
if [[ "${ENABLE_EVAL}" == "1" && "${ENABLE_OOD_EVAL}" == "1" ]]; then
    required_files+=("${OOD_EVAL_DATA}")
fi
for required_file in "${required_files[@]}"; do
    if [[ ! -f "${required_file}" ]]; then
        echo "Missing required file: ${required_file}" >&2
        if [[ "${required_file}" == "${MEGATRON_CHECKPOINT}/latest_checkpointed_iteration.txt" ]]; then
            echo "Set MEGATRON_CHECKPOINT to an existing converted model checkpoint." >&2
        fi
        exit 1
    fi
done

TOTAL_GPUS="${TOTAL_GPUS:-4}"
ACTOR_GPUS="${ACTOR_GPUS:-2}"
CRITIC_GPUS="${CRITIC_GPUS:-2}"
TP_SIZE="${TP_SIZE:-1}"
ROLLOUT_GPUS_PER_ENGINE="${ROLLOUT_GPUS_PER_ENGINE:-1}"
if (( TOTAL_GPUS != 4 || ACTOR_GPUS != 2 || CRITIC_GPUS != 2 )); then
    echo "The default topology is one 4-GPU node with 2 actor + 2 critic placement bundles." >&2
    echo "Got TOTAL_GPUS=${TOTAL_GPUS}, ACTOR_GPUS=${ACTOR_GPUS}, CRITIC_GPUS=${CRITIC_GPUS}." >&2
fi
if (( ACTOR_GPUS + CRITIC_GPUS > TOTAL_GPUS )); then
    echo "Insufficient Ray GPU resources: actor (${ACTOR_GPUS}) + critic (${CRITIC_GPUS}) exceeds TOTAL_GPUS (${TOTAL_GPUS})." >&2
    exit 2
fi
if (( ACTOR_GPUS % TP_SIZE != 0 )); then
    echo "ACTOR_GPUS must be divisible by TP_SIZE." >&2
    exit 2
fi
if (( TOTAL_GPUS % ROLLOUT_GPUS_PER_ENGINE != 0 )); then
    echo "TOTAL_GPUS must be divisible by ROLLOUT_GPUS_PER_ENGINE." >&2
    exit 2
fi

MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-4096}"
MAX_SEQUENCE_LEN="${MAX_SEQUENCE_LEN:-8192}"
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-8192}"
if (( MAX_RESPONSE_LEN >= MAX_SEQUENCE_LEN )); then
    echo "MAX_RESPONSE_LEN must be smaller than MAX_SEQUENCE_LEN so the prompt also fits." >&2
    exit 2
fi
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-64}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-8}"
NUM_STEPS_PER_ROLLOUT="${NUM_STEPS_PER_ROLLOUT:-2}"
NUM_ROLLOUT="${NUM_ROLLOUT:-250}"
NUM_CRITIC_ONLY_STEPS="${NUM_CRITIC_ONLY_STEPS:-20}"
ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-1.0}"
ACTOR_LR="${ACTOR_LR:-1e-6}"
CRITIC_LR="${CRITIC_LR:-4e-6}"
SEED="${SEED:-1234}"
ROLLOUT_SEED="${ROLLOUT_SEED:-42}"
EVAL_INTERVAL="${EVAL_INTERVAL:-10}"
EVAL_SAMPLES="${EVAL_SAMPLES:-1}"
BCE_VALUE_LOGIT_CLIP="${BCE_VALUE_LOGIT_CLIP:-0}"
VALUE_CLIP="${VALUE_CLIP:-0.2}"
ENABLE_WANDB="${ENABLE_WANDB:-1}"
ENABLE_DEBUG_DUMPS="${ENABLE_DEBUG_DUMPS:-0}"
ENABLE_POSITION_STATS="${ENABLE_POSITION_STATS:-1}"
MAX_LOG_POSITIONS="${MAX_LOG_POSITIONS:-512}"

PAIR_ID="${PAIR_ID:-$(date +%Y%m%d_%H%M%S)}"
EXP_NAME="${EXP_NAME:-Countdown-PPO}"
METHOD="${METHOD:-PPO}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${TPPO_ROOT}/outputs/countdown}"
SAVE_DIR="${SAVE_DIR:-${OUTPUT_ROOT}/checkpoints/${EXP_NAME}}"
RUN_DIR="${RUN_DIR:-${OUTPUT_ROOT}/runs/${EXP_NAME}}"
CRITIC_SAVE_DIR="${SAVE_DIR}/critic"

source "${PROJECT_ROOT}/scripts/models/qwen2.5-1.5B.sh"
MODEL_ARGS+=(
    --max-position-embeddings "${MAX_SEQUENCE_LEN}"
    --seq-length "${MAX_SEQUENCE_LEN}"
)

CKPT_ARGS=(
    --megatron-to-hf-mode bridge
    --hf-checkpoint "${HF_CHECKPOINT}"
    --ref-load "${MEGATRON_CHECKPOINT}"
    --load "${MEGATRON_CHECKPOINT}"
    --save "${SAVE_DIR}"
    --critic-save "${CRITIC_SAVE_DIR}"
    --save-interval "${EVAL_INTERVAL:-10}"
    --no-save-optim
)
if [[ "${ENABLE_EVAL:-1}" != "1" ]]; then
    echo "Best-checkpoint saving requires ENABLE_EVAL=1." >&2
    exit 2
fi
if [[ ! "${EVAL_INTERVAL:-10}" =~ ^[1-9][0-9]*$ ]]; then
    echo "EVAL_INTERVAL must be a positive integer for best-checkpoint saving." >&2
    exit 2
fi
CKPT_ARGS+=(
    --save-latest-best-ckpts
    --save-only-best
    --save-best-window-ends "${NUM_ROLLOUT}"
    --save-best-metric eval/countdown
)

ROLLOUT_ARGS=(
    --prompt-data "${TRAIN_DATA}"
    --input-key prompt
    --label-key label
    --apply-chat-template
    --rollout-shuffle
    --rollout-seed "${ROLLOUT_SEED}"
    --rm-type countdown
    --reward-key score
    --num-rollout "${NUM_ROLLOUT}"
    --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
    --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
    --rollout-max-context-len "${MAX_SEQUENCE_LEN}"
    --rollout-max-response-len "${MAX_RESPONSE_LEN}"
    --rollout-temperature "${ROLLOUT_TEMPERATURE}"
    --num-steps-per-rollout "${NUM_STEPS_PER_ROLLOUT}"
    --use-tis
    # --partial-rollout
    # --over-sampling-batch-size "$((ROLLOUT_BATCH_SIZE * 2))"
    --balance-data
)

EVAL_ARGS=()
if [[ "${ENABLE_EVAL}" == "1" ]]; then
    EVAL_DATASETS=(countdown "${EVAL_DATA}")
    if [[ "${ENABLE_OOD_EVAL}" == "1" ]]; then
        EVAL_DATASETS+=(countdown5_ood "${OOD_EVAL_DATA}")
    fi
    EVAL_ARGS=(
        --eval-interval "${EVAL_INTERVAL}"
        --eval-prompt-data "${EVAL_DATASETS[@]}"
        --n-samples-per-eval-prompt "${EVAL_SAMPLES}"
        --eval-max-context-len "${MAX_SEQUENCE_LEN}"
        --eval-max-response-len "${MAX_RESPONSE_LEN}"
        --eval-temperature 1.0
        --eval-top-p 1.0
        --log-passrate
    )
fi

PERF_ARGS=(
    --tensor-model-parallel-size "${TP_SIZE}"
    --pipeline-model-parallel-size 1
    --context-parallel-size 1
    --recompute-method uniform
    --recompute-num-layers 1
    --use-dynamic-batch-size
    --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"
    --transformer-impl transformer_engine
    --bf16
)
if (( TP_SIZE > 1 )); then
    PERF_ARGS+=(--sequence-parallel)
fi

PPO_ARGS=(
    --advantage-estimator ppo
    --kl-loss-coef 0.0
    --kl-loss-type low_var_kl
    --kl-coef 0.0
    --eps-clip 0.2
    --eps-clip-high 0.2
    --num-critic-only-steps "${NUM_CRITIC_ONLY_STEPS}"
    --critic-lr "${CRITIC_LR}"
    --critic-num-nodes 1
    --critic-num-gpus-per-node "${CRITIC_GPUS}"
    --value-clip "${VALUE_CLIP}"
)
PPO_ARGS+=(--critic-output-layer-init-mode legacy_v024)
if [[ "${VALUE_LOSS}" == "bce" ]]; then
    PPO_ARGS+=(
        --use-bce-value-loss
        --bce-value-logit-clip "${BCE_VALUE_LOGIT_CLIP}"
    )
fi

OPTIMIZER_ARGS=(
    --optimizer adam
    --lr "${ACTOR_LR}"
    --lr-decay-style constant
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.98
    --use-precision-aware-optimizer
)

WANDB_ARGS=()
if [[ "${ENABLE_WANDB}" == "1" ]]; then
    export WANDB_DIR="${RUN_DIR}/wandb"
    export WANDB_MODE="${WANDB_MODE:-offline}"
    WANDB_ARGS=(
        --use-wandb
        --wandb-project "${METHOD}"
        --wandb-group "${EXP_NAME}"
        --disable-wandb-random-suffix
        --wandb-dir "${WANDB_DIR}"
        --wandb-mode "${WANDB_MODE}"
    )
    if [[ -n "${WANDB_API_KEY:-}" ]]; then
        WANDB_ARGS+=(--wandb-key "${WANDB_API_KEY}")
    fi
fi

SGLANG_ARGS=(
    --num-gpus-per-node "${TOTAL_GPUS}"
    --rollout-num-gpus-per-engine "${ROLLOUT_GPUS_PER_ENGINE}"
    --sglang-context-length "${MAX_SEQUENCE_LEN}"
    --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC:-0.65}"
    --sglang-cuda-graph-bs 1 2 4 8 $(seq 16 8 256)
)

TRAIN_RESOURCE_ARGS=(
    --actor-num-nodes 1
    --actor-num-gpus-per-node "${ACTOR_GPUS}"
    --colocate
)

MISC_ARGS=(
    --seed "${SEED}"
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
    --attention-backend flash
    --lambd 1
)
if [[ "${ENABLE_POSITION_STATS}" == "1" ]]; then
    MISC_ARGS+=(
        --log-position-value-stats
        --log-position-advantage-stats
        --max-log-positions "${MAX_LOG_POSITIONS}"
        --max-log-positions-advantage "${MAX_LOG_POSITIONS}"
    )
fi

DEBUG_ARGS=()
if [[ "${ENABLE_DEBUG_DUMPS}" == "1" ]]; then
    mkdir -p "${RUN_DIR}/debug_data"
    DEBUG_ARGS+=(
        --dump-details "${RUN_DIR}/debug_data"
    )
fi

NVLINK_COUNT="$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l || true)"
if (( NVLINK_COUNT > 0 )); then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
export PYTHONBUFFERED=16

export PYTHONPATH="${MEGATRON_LM_PATH}:${PROJECT_ROOT}:${PYTHONPATH:-}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-${HAS_NVLINK}}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
# Qwen2.5-Math-1.5B declares a 4096-token context. The experiment requests
# 4096 new tokens in addition to the prompt, so SGLang must explicitly allow
# the small RoPE extrapolation beyond the checkpoint metadata limit.
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN="${SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN:-1}"

RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-26500}"
RAY_STARTED=0
cleanup() {
    local status=$?
    trap - EXIT INT TERM
    if [[ "${RAY_STARTED}" == "1" ]]; then
        ray stop --force >/dev/null 2>&1 || true
    fi
    exit "${status}"
}
trap cleanup EXIT INT TERM

cd "${PROJECT_ROOT}"
echo "method=${METHOD} value_loss=${VALUE_LOSS} exp=${EXP_NAME} GPUs=${TOTAL_GPUS} TP=${TP_SIZE}"
echo "rollout=${ROLLOUT_BATCH_SIZE}x${N_SAMPLES_PER_PROMPT}, response_len=${MAX_RESPONSE_LEN}, lambda=1"
echo "save=${SAVE_DIR}"

TRAIN_COMMAND=(
    python3 "${PROJECT_ROOT}/train.py"
    "${MODEL_ARGS[@]}"
    "${TRAIN_RESOURCE_ARGS[@]}"
    "${CKPT_ARGS[@]}"
    "${ROLLOUT_ARGS[@]}"
    "${OPTIMIZER_ARGS[@]}"
    "${PPO_ARGS[@]}"
    "${WANDB_ARGS[@]}"
    "${PERF_ARGS[@]}"
    "${EVAL_ARGS[@]}"
    "${SGLANG_ARGS[@]}"
    "${DEBUG_ARGS[@]}"
    "${MISC_ARGS[@]}"
)
if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf 'DRY RUN command:'
    printf ' %q' "${TRAIN_COMMAND[@]}"
    printf '\n'
    exit 0
fi
if [[ "${VALIDATE_ARGS:-0}" == "1" ]]; then
    python3 -c \
        'from slime.utils.arguments import parse_args; from slime.utils.add_my_custom_arguments import add_my_custom_arguments; args = parse_args(add_custom_arguments=add_my_custom_arguments); print("arguments_valid", args.advantage_estimator, args.lambd)' \
        "${TRAIN_COMMAND[@]:2}"
    exit 0
fi

if ray status >/dev/null 2>&1; then
    echo "A Ray runtime is already active. Stop it first, or use a dedicated worker." >&2
    exit 1
fi

mkdir -p "${SAVE_DIR}" "${CRITIC_SAVE_DIR}" "${RUN_DIR}/wandb"

ray start --head \
    --node-ip-address "${MASTER_ADDR}" \
    --num-gpus "${TOTAL_GPUS}" \
    --disable-usage-stats \
    --dashboard-host 0.0.0.0 \
    --dashboard-port "${RAY_DASHBOARD_PORT}"
RAY_STARTED=1

RUNTIME_ENV_JSON="{\"env_vars\":{\"PYTHONPATH\":\"${MEGATRON_LM_PATH}:${PROJECT_ROOT}\",\"CUDA_DEVICE_MAX_CONNECTIONS\":\"1\",\"NCCL_NVLS_ENABLE\":\"${NCCL_NVLS_ENABLE}\",\"SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN\":\"${SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN}\"}}"
LOG_FILE="${RUN_DIR}/${EXP_NAME}.log"

ray job submit --address="http://127.0.0.1:${RAY_DASHBOARD_PORT}" \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- "${TRAIN_COMMAND[@]}" 2>&1 | tee -a "${LOG_FILE}"
