#!/bin/bash

if [[ "${ALLOW_GLOBAL_RUNTIME_CLEANUP:-0}" == "1" ]]; then
    pkill -9 sglang || true
    sleep 3
    ray stop --force || true
    pkill -9 ray || true
    sleep 3
    pkill -9 ray || true
fi

set -ex
set -o pipefail


NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l || true)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi



export PYTHONBUFFERED=16
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
TPPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
SLIME_ROOT="${TPPO_ROOT}/slime"
MEGATRON_LM_PATH="${MEGATRON_LM_PATH:-/path/to/Megatron-LM}"
export TPPO_ROOT SLIME_ROOT MEGATRON_LM_PATH
export PYTHONPATH="${SLIME_ROOT}:${MEGATRON_LM_PATH}:${PYTHONPATH:-}"
PPO_BASELINE_ROOT="${SLIME_ROOT}"
export PYTHONPATH=$PPO_BASELINE_ROOT:$PYTHONPATH
cd $PPO_BASELINE_ROOT


echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"
METHOD="${METHOD:-PPO}"
EXP_NAME="${EXP_NAME:-8B-Base-PPO-Polaris}"
CRITIC_OUTPUT_LAYER_INIT_MODE="${CRITIC_OUTPUT_LAYER_INIT_MODE:-legacy_v024}"
CRITIC_ZERO_INIT_OUTPUT_LAYER="${CRITIC_ZERO_INIT_OUTPUT_LAYER:-0}"


MODEL_CONFIG_PATH="${MODEL_CONFIG_PATH:-${TPPO_ROOT}/scripts/models/qwen3-8B-Base.sh}"
source "${MODEL_CONFIG_PATH}"
WANDB_RUN_NAME="${EXP_NAME}_$(date +%Y%m%d_%H%M%S)"
SAVE_DIR="${SAVE_DIR:-${TPPO_ROOT}/outputs/checkpoints/${METHOD}/${EXP_NAME}/}"
LOG_DIR="${LOG_DIR:-${TPPO_ROOT}/outputs/logs}"
EXP_RUN_DIR="${LOG_DIR}/${METHOD}/${EXP_NAME}"
mkdir -p "${EXP_RUN_DIR}" "${EXP_RUN_DIR}/wandb"
CRITIC_SAVE_DIR="${SAVE_DIR}/critic"
CRITIC2_SAVE_DIR="${SAVE_DIR}/critic2"
   mkdir -p "${SAVE_DIR}" "${CRITIC_SAVE_DIR}" "${CRITIC2_SAVE_DIR}"
GPU_COUNT="${GPU_COUNT:-8}"
ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-1}"
ACTOR_GPU_COUNT="${ACTOR_GPU_COUNT:-4}"
CRITIC_NUM_NODES="${CRITIC_NUM_NODES:-1}"
CRITIC_GPU_COUNT="${CRITIC_GPU_COUNT:-$ACTOR_GPU_COUNT}"
TP_SIZE="${TP_SIZE:-$ACTOR_GPU_COUNT}"
PP_SIZE="${PP_SIZE:-1}"
CP_SIZE="${CP_SIZE:-1}"
EP_SIZE="${EP_SIZE:-1}"
ETP_SIZE="${ETP_SIZE:-1}"
MAX_LEN=$((1024 * 8))
MAX_TOKENS_PER_GPU=$((($MAX_LEN / $CP_SIZE) + 1024))
ROLLOUT_BATCH_SIZE=64
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-8}"
NUM_STEPS_PER_ROLLOUT=2
NUM_ROLLOUT="${NUM_ROLLOUT:-520}"
NUM_CRITIC_ONLY_STEPS="${NUM_CRITIC_ONLY_STEPS:-20}"
PPO_ACTOR_SEGMENT_RATIOS="${PPO_ACTOR_SEGMENT_RATIOS:-}"
PPO_NORMALIZE_ADVANTAGES="${PPO_NORMALIZE_ADVANTAGES:-0}"
OVER_SAMPLING_BATCH_SIZE=$((ROLLOUT_BATCH_SIZE * 2))
START_ROLLOUT_ARG=()

# Check if resume checkpoint exists
RESUME_CHECKPOINT_DIR="${RESUME_CHECKPOINT_DIR:-}"
RESUME_CRITIC_CHECKPOINT_DIR="${RESUME_CRITIC_CHECKPOINT_DIR:-${RESUME_CHECKPOINT_DIR%/}/critic}"
RESUME_ROLLOUT_ID="${RESUME_ROLLOUT_ID:-}"
if [ -n "$RESUME_CHECKPOINT_DIR" ] && [ -d "$RESUME_CHECKPOINT_DIR" ]; then
    # Check for latest_checkpointed_iteration.txt or similar checkpoint indicator
    if [ -f "${RESUME_CHECKPOINT_DIR}/latest_checkpointed_iteration.txt" ]; then
        LATEST_ITER=$(cat "${RESUME_CHECKPOINT_DIR}/latest_checkpointed_iteration.txt" | tr -d '[:space:]')
        RESUME_ROLLOUT_ID="${RESUME_ROLLOUT_ID:-$((LATEST_ITER + 1))}"
        echo "✅ Found resume checkpoint at: ${RESUME_CHECKPOINT_DIR}"
        echo "   Latest iteration: ${LATEST_ITER}"
        echo "   Will resume from rollout ${RESUME_ROLLOUT_ID}"
        USE_RESUME_CHECKPOINT=true
    elif [ -n "$(find "${RESUME_CHECKPOINT_DIR}" -name "iter_*" -type d | head -1)" ]; then
        # Alternative: check if any iteration directory exists
        LATEST_ITER="$(find "${RESUME_CHECKPOINT_DIR}" -maxdepth 1 -type d -name 'iter_*' -printf '%f\n' 2>/dev/null \
            | awk -F_ '/^iter_[0-9]+$/ { n=$2+0; if (!seen || n>max) { max=n; seen=1 } } END { if (seen) print max }')"
        RESUME_ROLLOUT_ID="${RESUME_ROLLOUT_ID:-$((LATEST_ITER + 1))}"
        echo "✅ Found resume checkpoint at: ${RESUME_CHECKPOINT_DIR}"
        echo "   Will resume from rollout ${RESUME_ROLLOUT_ID}"
        USE_RESUME_CHECKPOINT=true
    else
        echo "⚠️  Resume checkpoint directory exists but no checkpoint found, starting from initial checkpoint"
        USE_RESUME_CHECKPOINT=false
    fi
else
    echo "⚠️  Resume checkpoint not found or not set, starting from initial checkpoint"
    USE_RESUME_CHECKPOINT=false
fi

# Check critic checkpoint if resume is enabled
if [ "$USE_RESUME_CHECKPOINT" = true ] && [ -n "$RESUME_CRITIC_CHECKPOINT_DIR" ] && [ -d "$RESUME_CRITIC_CHECKPOINT_DIR" ]; then
    echo "✅ Found resume critic checkpoint at: ${RESUME_CRITIC_CHECKPOINT_DIR}"
    USE_RESUME_CRITIC_CHECKPOINT=true
else
    USE_RESUME_CRITIC_CHECKPOINT=false
fi

# Check critic2 checkpoint if resume is enabled
if [ "$USE_RESUME_CHECKPOINT" = true ] && [ -n "$RESUME_CRITIC2_CHECKPOINT_DIR" ] && [ -d "$RESUME_CRITIC2_CHECKPOINT_DIR" ]; then
    echo "✅ Found resume critic2 checkpoint at: ${RESUME_CRITIC2_CHECKPOINT_DIR}"
    USE_RESUME_CRITIC2_CHECKPOINT=true
else
    USE_RESUME_CRITIC2_CHECKPOINT=false
fi
# ========================================================================

# Set checkpoint arguments based on resume status
if [ "$USE_RESUME_CHECKPOINT" = true ]; then
    CKPT_ARGS=(
       --megatron-to-hf-mode bridge
       --hf-checkpoint "${HF_MODEL_PATH:-/path/to/models/Qwen3-8B-Base/}"
       --ref-load "${MEGATRON_MODEL_PATH:-/path/to/models/Qwen3-8B-Base-megatron}"
       --load "${RESUME_CHECKPOINT_DIR}"
       --start-rollout-id "${RESUME_ROLLOUT_ID}"
       --save "${SAVE_DIR}"
       --critic-save "${CRITIC_SAVE_DIR}"
       --save-interval "${EVAL_INTERVAL:-10}"
       --no-load-optim
       --no-load-rng
    )
    if [ "$USE_RESUME_CRITIC_CHECKPOINT" = true ]; then
       CKPT_ARGS+=(--critic-load "${RESUME_CRITIC_CHECKPOINT_DIR}")
    fi
else
    CKPT_ARGS=(
       --megatron-to-hf-mode bridge
       --hf-checkpoint "${HF_MODEL_PATH:-/path/to/models/Qwen3-8B-Base/}"
       --ref-load "${MEGATRON_MODEL_PATH:-/path/to/models/Qwen3-8B-Base-megatron}"
       --load "${MEGATRON_MODEL_PATH:-/path/to/models/Qwen3-8B-Base-megatron}"
       --save ${SAVE_DIR}
       --critic-save ${CRITIC_SAVE_DIR}
       --save-interval "${EVAL_INTERVAL:-10}"
       --no-save-optim
    )
fi


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
    --save-best-metric eval/overall
)

ROLLOUT_ARGS=(
   --prompt-data "${TPPO_ROOT}/data/polaris/qwen3-4b-s1-polaris-p1-format.jsonl"
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type p1
   --rm-url "${RM_URL:-http://127.0.0.1:8001/}"
   --reward-key score
   
   --num-rollout $NUM_ROLLOUT
   ${START_ROLLOUT_ARG[@]}
   --rollout-batch-size $ROLLOUT_BATCH_SIZE
   --n-samples-per-prompt $N_SAMPLES_PER_PROMPT
   --rollout-max-response-len $MAX_LEN
   --rollout-temperature 1.0

   # --global-batch-size 128
   --num-steps-per-rollout $NUM_STEPS_PER_ROLLOUT
   --over-sampling-batch-size $OVER_SAMPLING_BATCH_SIZE
   --partial-rollout
   --over-sampling-batch-size $((ROLLOUT_BATCH_SIZE * 2))
   --balance-data
   # --dynamic-sampling-filter-path slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std
)
   ROLLOUT_ARGS+=(--use-tis)
N_SAMPLES_PER_EVAL=(4 4 4 4 1 1)
# N_SAMPLES_PER_EVAL=(32 32 32 32 1 1)
IMO_PATH=/path/to/test_jsonl
EVAL_ARGS=(
   --eval-interval "${EVAL_INTERVAL:-10}"
   --eval-prompt-data aime24 ${TPPO_ROOT}/data/eval/aime_2024.jsonl aime25 ${TPPO_ROOT}/data/eval/aime_2025.jsonl aime26 ${TPPO_ROOT}/data/eval/aime_2026.jsonl amc23 ${TPPO_ROOT}/data/eval/amc23.jsonl math500 ${TPPO_ROOT}/data/eval/math500.jsonl minervamath ${TPPO_ROOT}/data/eval/minervamath.jsonl
   --n-samples-per-eval-prompt ${N_SAMPLES_PER_EVAL[@]}
   --eval-max-response-len $((MAX_LEN * 3))
   --eval-top-p 0.95
   --eval-temperature 1.0
   --log-passrate
)


PERF_ARGS=(
   --tensor-model-parallel-size $TP_SIZE
   --sequence-parallel
   --pipeline-model-parallel-size $PP_SIZE
   --context-parallel-size $CP_SIZE
   --expert-model-parallel-size $EP_SIZE
   --expert-tensor-parallel-size $ETP_SIZE

   # --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   # --micro-batch-size 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu $MAX_TOKENS_PER_GPU
   --transformer-impl transformer_engine
   --bf16
   # --fp8-format e4m3
   # --fp8-recipe blockwise
)

# GRPO_ARGS=(
#    --advantage-estimator ppo
#    --use-kl-loss
#    --kl-loss-coef 0.00
#    --kl-loss-type low_var_kl
#    --entropy-coef 0.00
#    --eps-clip 0.2
#    --eps-clip-high 0.28
# )

PPO_ARGS=(
   --advantage-estimator "${ADVANTAGE_ESTIMATOR:-ppo}"
   # --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --kl-coef 0.00
#    --entropy-coef 0.00
   --eps-clip 0.2
   # --eps-clip-high 0.28
   --eps-clip-high 0.2
   --num-critic-only-steps "$NUM_CRITIC_ONLY_STEPS"
   # --num-critic-only-steps 1
   # --normalize-advantages
   --critic-lr 4e-6
   --critic-num-nodes $CRITIC_NUM_NODES
   --critic-num-gpus-per-node $CRITIC_GPU_COUNT


   # Critic2 configuration (same settings as critic1)
   # --critic2-num-nodes 1
   # --critic2-num-gpus-per-node 2
   # --critic2-lr 4e-6
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98

   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
)

export WANDB_DIR="${EXP_RUN_DIR}/wandb"
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_MODE="offline"

WANDB_ARGS=(
   --use-wandb
   --wandb-project "${METHOD}"
   --wandb-group "${WANDB_RUN_NAME}"
   --disable-wandb-random-suffix
   --wandb-dir ${WANDB_DIR}
   --wandb-mode ${WANDB_MODE}
)


SGLANG_ARGS=(
   --rollout-num-gpus-per-engine "${ROLLOUT_NUM_GPUS_PER_ENGINE:-2}"
   --sglang-mem-fraction-static 0.7
   --sglang-cuda-graph-bs 1 2 4 8 $(seq 16 8 256)
   # --sglang-speculative-algorithm EAGLE3
   # --sglang-speculative-num-steps 3
   # --sglang-speculative-eagle-topk 1
   # --sglang-speculative-num-draft-tokens 4
)
# export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
MISC_ARGS=(
   # default dropout in megatron is 0.1
   --attention-dropout 0.0
   --hidden-dropout 0.0
   # should be good for model performance
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   # need to comment this when using model with MLA
   --attention-backend flash
)


DEBUG_DIR="${EXP_RUN_DIR}/debug_data"
ROLLOUT_DEBUG_DIR="${DEBUG_DIR}/rollout"
TRAIN_DEBUG_DIR="${DEBUG_DIR}/train"
ANALYSIS_DIR="${DEBUG_DIR}/analysis"
mkdir -p "${ROLLOUT_DEBUG_DIR}" "${TRAIN_DEBUG_DIR}" "${ANALYSIS_DIR}"
echo "Debug data will be saved to: ${DEBUG_DIR}"

DEBUG_ARGS=(
   --save-debug-rollout-data "${ROLLOUT_DEBUG_DIR}/rollout_data_{}.pkl"
   --save-debug-train-data "${TRAIN_DEBUG_DIR}/train_data_{}.pkl"
   --dump-details "${ANALYSIS_DIR}/training_details.json"
)

CUSTOM_ARGS=(
#    # --use-asyppo
#    # --use-hf-config-for-megatron
#    --log-position-value-stats
#    --max-log-positions 500
   --critic-output-layer-init-mode "${CRITIC_OUTPUT_LAYER_INIT_MODE}"
   --lambd 1
#    --lambd-actor 1
#    # --lambd-critic 1
#    --lambd-critic 1
#    # --use-adaptive-lambda
#    # --alpha 0.05
#    --log-position-advantage-stats
#    # --use-positive-nll-loss
# #    --positive-nll-coef 0.1
# #    --positive-reward-threshold 0.0
   --eval-first
#    --finetune
#    --no-load-optim
#    # --use-asytrain-critic
#    # --use-advantage-diff-mask
#    # --advantage-diff-mask-k 0.1
#    # --use-entropy-value-divergence-filter
#    # --entropy-divergence-filter-h 0.2
#    # --entropy-coef 0.001
# #    --critic-num-gpus-per-node 2
# #    --critic-num-nodes 1
# #    --critic2-num-gpus-per-node 2
# #    --critic2-num-nodes 1
#    # --eval-use-xverify
#    # --eval-group
#    --log-probs-chunk-size 8192
#    # --train-use-xverify
#    # --eval-log-dir ${DEBUG_DIR}/eval
#    # --start-rollout-id 0
#    # --critic-token-loss
#    # --critic-token 1699
#    # --critic-token 198
#    # --critic-token-interval 32
   # --policy-token-value-broadcast
#    --policy-token 198
   # --policy-token-interval 32
)
if [[ "${PPO_USE_DECOUPLED_GAE:-0}" == "1" ]]; then
   CUSTOM_ARGS+=(
      --use-decoupled-gae
      --lambd-actor "${PPO_LAMBD_ACTOR:-0.95}"
      --lambd-critic "${PPO_LAMBD_CRITIC:-1}"
   )
fi
if [[ -n "${PPO_ACTOR_SEGMENT_RATIOS}" ]]; then
   read -r -a PPO_ACTOR_SEGMENT_RATIO_ARGS <<< "${PPO_ACTOR_SEGMENT_RATIOS}"
   CUSTOM_ARGS+=(--ppo-actor-segment-ratios "${PPO_ACTOR_SEGMENT_RATIO_ARGS[@]}")
fi
if [[ "${PPO_NORMALIZE_ADVANTAGES}" == "1" ]]; then
   PPO_ARGS+=(--normalize-advantages)
fi
if [[ "${CRITIC_ZERO_INIT_OUTPUT_LAYER}" == "1" ]]; then
   CUSTOM_ARGS+=(--critic-zero-init-output-layer)
fi


# Launch the Ray head on rank 0 and join all other manually started nodes.
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export NODE_RANK="${NODE_RANK:-0}"
export NNODES="${NNODES:-1}"
export GPUS_PER_NODE="${GPUS_PER_NODE:-$GPU_COUNT}"
export RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-26500}"
start_or_join_manual_ray_cluster() {
    local node_rank="${NODE_RANK:-0}"
    local num_nodes="${NNODES:-1}"
    local gpus_per_node="${GPUS_PER_NODE:-8}"
    local ray_port="${RAY_PORT:-6379}"
    local dashboard_port="${RAY_DASHBOARD_PORT:-26500}"
    local node_ip="${NODE_IP_ADDR:-}"

    if [[ -z "${MASTER_ADDR:-}" ]]; then
        echo "MASTER_ADDR must be set for a manual Ray cluster." >&2
        return 1
    fi
    if [[ -z "${node_ip}" ]]; then
        node_ip="$(hostname -I | awk '{print $1}')"
    fi
    if [[ -z "${node_ip}" ]]; then
        echo "Failed to determine this node's IP address." >&2
        return 1
    fi

    if [[ "${node_rank}" == "0" ]]; then
        ray start --head \
            --node-ip-address="${MASTER_ADDR}" \
            --port="${ray_port}" \
            --num-gpus="${gpus_per_node}" \
            --disable-usage-stats \
            --dashboard-host=0.0.0.0 \
            --dashboard-port="${dashboard_port}"

        local expected_gpus=$((num_nodes * gpus_per_node))
        local attempt available_gpus
        for attempt in $(seq 1 "${RAY_CLUSTER_WAIT_ATTEMPTS:-120}"); do
            available_gpus="$(python3 - "${MASTER_ADDR}:${ray_port}" <<'PY'
import sys
import ray

try:
    ray.init(address=sys.argv[1], logging_level="ERROR")
    print(int(ray.cluster_resources().get("GPU", 0)))
except Exception:
    print(0)
finally:
    if ray.is_initialized():
        ray.shutdown()
PY
)"
            if [[ "${available_gpus}" =~ ^[0-9]+$ ]] && (( available_gpus >= expected_gpus )); then
                echo "Ray cluster is ready: ${available_gpus}/${expected_gpus} GPUs."
                return 0
            fi
            echo "Waiting for Ray workers (${available_gpus:-0}/${expected_gpus} GPUs, attempt ${attempt})..."
            sleep "${RAY_CLUSTER_WAIT_SECONDS:-5}"
        done
        echo "Timed out waiting for ${expected_gpus} GPUs to join the Ray cluster." >&2
        return 1
    fi

    local attempt
    for attempt in $(seq 1 "${RAY_WORKER_JOIN_ATTEMPTS:-120}"); do
        if ray start \
            --address="${MASTER_ADDR}:${ray_port}" \
            --node-ip-address="${node_ip}" \
            --num-gpus="${gpus_per_node}" \
            --disable-usage-stats \
            --block; then
            return 0
        fi
        echo "Ray head is not ready; retrying worker join (${attempt})..."
        sleep "${RAY_WORKER_JOIN_SECONDS:-5}"
    done
    echo "Worker rank ${node_rank} failed to join Ray at ${MASTER_ADDR}:${ray_port}." >&2
    return 1
}
start_or_join_manual_ray_cluster

# Build the runtime environment JSON with proper variable substitution
# Slime comes only from this project checkout; Megatron is a separate
# dependency required by slime.backends.megatron_utils.
SLIME_RUNTIME_PYTHONPATH="${PPO_BASELINE_ROOT}:${MEGATRON_LM_PATH}/"
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${SLIME_RUNTIME_PYTHONPATH}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\"
  }
}"

mkdir -p "${EXP_RUN_DIR}"
LOG_FILE="${EXP_RUN_DIR}/${EXP_NAME}.log"
echo "Log file will be saved to: ${LOG_FILE}"

export RANK="${NODE_RANK}"
export NODE_COUNT="${NNODES}"
export PROC_PER_NODE="${PROC_PER_NODE:-$GPUS_PER_NODE}"

if [ -z "$RANK" ]; then
  echo "RANK not set. Defaulting to RANK=0 for single node"
  export RANK=0
fi


# ========= Submit Ray Job (rank 0 only) =========
if [ "$RANK" == "0" ]; then
   ray job submit --address="http://127.0.0.1:${RAY_DASHBOARD_PORT}" \
      --runtime-env-json="${RUNTIME_ENV_JSON}" \
      -- python3 ${PPO_BASELINE_ROOT}/train.py \
      ${MODEL_ARGS[@]} \
      --actor-num-nodes $ACTOR_NUM_NODES \
      --actor-num-gpus-per-node $ACTOR_GPU_COUNT \
      --colocate \
      ${CKPT_ARGS[@]} \
      ${ROLLOUT_ARGS[@]} \
      ${OPTIMIZER_ARGS[@]} \
      ${PPO_ARGS[@]} \
      ${WANDB_ARGS[@]} \
      ${PERF_ARGS[@]} \
      ${EVAL_ARGS[@]} \
      ${SGLANG_ARGS[@]} \
      ${DEBUG_ARGS[@]} \
      ${CUSTOM_ARGS[@]} \
      ${MISC_ARGS[@]} 2>&1 | tee -a "${LOG_FILE}"
fi
