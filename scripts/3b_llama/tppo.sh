#!/bin/bash

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
TPPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
SLIME_ROOT="${TPPO_ROOT}/slime"
MEGATRON_LM_PATH="${MEGATRON_LM_PATH:-/path/to/Megatron-LM}"
export TPPO_ROOT SLIME_ROOT MEGATRON_LM_PATH
export PYTHONPATH="${SLIME_ROOT}:${MEGATRON_LM_PATH}:${PYTHONPATH:-}"

export METHOD="${METHOD:-TPPO}"
export EXP_NAME="${EXP_NAME:-3B-Llama32-TPPO-Polaris}"
export ENABLE_CRITIC_TPPO="${ENABLE_CRITIC_TPPO:-1}"
export CRITIC_DELTA_SCALE="${CRITIC_DELTA_SCALE:-1.0}"
export CRITIC_TPPO_HEAD_SPACE="${CRITIC_TPPO_HEAD_SPACE:-0}"
export CRITIC_ZERO_INIT_OUTPUT_LAYER="${CRITIC_ZERO_INIT_OUTPUT_LAYER:-1}"

# PPO training script for llama3.2-3B-Instruct
# Adapted from 8B baseline for 3B model

# Optional cleanup for a dedicated training worker.
if [[ "${ALLOW_GLOBAL_RUNTIME_CLEANUP:-0}" == "1" ]]; then
    pkill -9 sglang || true
    ray stop --force || true
    pkill -9 ray || true
fi

set -ex
set -o pipefail

is_truthy() {
   case "$1" in
      1|true|TRUE|yes|YES|on|ON) return 0 ;;
      *) return 1 ;;
   esac
}


NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l || true)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi



export PYTHONBUFFERED=16
PPO_BASELINE_ROOT="${SLIME_ROOT}"
export PYTHONPATH=$PPO_BASELINE_ROOT:$PYTHONPATH
cd $PPO_BASELINE_ROOT


echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"


ENABLE_CRITIC_TPPO="${ENABLE_CRITIC_TPPO:-0}"
CRITIC_DELTA_SCALE="${CRITIC_DELTA_SCALE:-1.0}"
CRITIC_TPPO_HEAD_SPACE="${CRITIC_TPPO_HEAD_SPACE:-0}"
CRITIC_SEPARATE_V0_HEAD="${CRITIC_SEPARATE_V0_HEAD:-0}"
CRITIC_SEPARATE_V0_HEAD_LAST_TOKEN_ONLY="${CRITIC_SEPARATE_V0_HEAD_LAST_TOKEN_ONLY:-0}"
CRITIC_ZERO_INIT_OUTPUT_LAYER="${CRITIC_ZERO_INIT_OUTPUT_LAYER:-0}"

# Load llama3.2-3B model configuration
source "${PPO_BASELINE_ROOT}/scripts/models/llama3.2-3B-Instruct.sh"

WANDB_RUN_NAME="${EXP_NAME}_$(date +%Y%m%d_%H%M%S)"
SAVE_DIR="${SAVE_DIR:-${TPPO_ROOT}/outputs/checkpoints/${METHOD}/${EXP_NAME}/}"
LOG_DIR="${LOG_DIR:-${TPPO_ROOT}/outputs/logs}"
EXP_RUN_DIR="${LOG_DIR}/${METHOD}/${EXP_NAME}"
mkdir -p "${EXP_RUN_DIR}" "${EXP_RUN_DIR}/wandb"
CRITIC_SAVE_DIR="${SAVE_DIR}/critic"
CRITIC2_SAVE_DIR="${SAVE_DIR}/critic2"
   mkdir -p "${SAVE_DIR}" "${CRITIC_SAVE_DIR}" "${CRITIC2_SAVE_DIR}"

# ============================================================================
# Resource Configuration - Optimized for 3B model
# ============================================================================
# 3B model is smaller, so we can use fewer GPUs or increase batch size
GPU_COUNT=4              # Total GPUs available
ACTOR_GPU_COUNT=2        # 3B only needs 2 GPUs with TP (vs 4 for 8B)
TP_SIZE=$ACTOR_GPU_COUNT # Tensor Parallelism size
PP_SIZE=1                # Pipeline Parallelism
CP_SIZE=1                # Context Parallelism
EP_SIZE=1                # Expert Parallelism (for MoE models)
ETP_SIZE=1               # Expert Tensor Parallelism

MAX_LEN=$((1024 * 8))    # 8K max response length
MAX_TOKENS_PER_GPU=$((($MAX_LEN / $CP_SIZE) + 1024))

# 3B model has less memory footprint, can potentially increase batch size
ROLLOUT_BATCH_SIZE=128   # Increased from 64 (8B) to 128 (3B)
N_SAMPLES_PER_PROMPT=8
NUM_STEPS_PER_ROLLOUT=2
NUM_ROLLOUT="${NUM_ROLLOUT:-520}"
START_ROLLOUT_ARG=()

# ============================================================================
# Resume Checkpoint Logic
# ============================================================================
if [ -n "$RESUME_CHECKPOINT_DIR" ] && [ -d "$RESUME_CHECKPOINT_DIR" ]; then
    # Check for latest_checkpointed_iteration.txt or similar checkpoint indicator
    if [ -f "${RESUME_CHECKPOINT_DIR}/latest_checkpointed_iteration.txt" ]; then
        LATEST_ITER=$(cat "${RESUME_CHECKPOINT_DIR}/latest_checkpointed_iteration.txt" | tr -d '[:space:]')
        echo "✅ Found resume checkpoint at: ${RESUME_CHECKPOINT_DIR}"
        echo "   Latest iteration: ${LATEST_ITER}"
        echo "   Will resume from rollout ${RESUME_ROLLOUT_ID}"
        USE_RESUME_CHECKPOINT=true
    elif [ -n "$(find "${RESUME_CHECKPOINT_DIR}" -name "iter_*" -type d | head -1)" ]; then
        # Alternative: check if any iteration directory exists
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

# ============================================================================
# Checkpoint Configuration
# ============================================================================
# TODO: Update these paths to your actual llama3.2-3B checkpoint locations
LLAMA32_3B_HF_PATH="${HF_MODEL_PATH:-/path/to/models/llama3.2-3B-Instruct}"
LLAMA32_3B_MEGATRON_PATH="${MEGATRON_MODEL_PATH:-/path/to/models/llama3.2-3B-Instruct-megatron}"

# Set checkpoint arguments based on resume status
if [ "$USE_RESUME_CHECKPOINT" = true ]; then
    CKPT_ARGS=(
      #  --hf-checkpoint ${LLAMA32_3B_HF_PATH}
      #  --ref-load ${LLAMA32_3B_MEGATRON_PATH}
      #  --load ${RESUME_CHECKPOINT_DIR}
      #  --start-rollout-id ${RESUME_ROLLOUT_ID}
      #  --save ${RESUME_CHECKPOINT_DIR}

      #  $(if [ "$USE_RESUME_CRITIC_CHECKPOINT" = true ]; then echo "--critic-load ${RESUME_CRITIC_CHECKPOINT_DIR}"; fi)
      #  --critic-save ${CRITIC_SAVE_DIR}

      #  --save-interval 500
    )
else
    CKPT_ARGS=(
       --megatron-to-hf-mode bridge
       --hf-checkpoint ${LLAMA32_3B_HF_PATH}
       --ref-load ${LLAMA32_3B_MEGATRON_PATH}
       --load ${LLAMA32_3B_MEGATRON_PATH}
       --save ${SAVE_DIR}
       --critic-save ${CRITIC_SAVE_DIR}
       --save-interval "${EVAL_INTERVAL:-10}"
       --no-save-optim
    )
fi


# ============================================================================
# Rollout Configuration
# ============================================================================
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
   --prompt-data ${TPPO_ROOT}/data/polaris/qwen3-4b-s1-polaris-p1-format.jsonl
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

   --num-steps-per-rollout $NUM_STEPS_PER_ROLLOUT
   --use-tis
   --partial-rollout
   --over-sampling-batch-size $((ROLLOUT_BATCH_SIZE * 2))
   --balance-data
)

# ============================================================================
# Evaluation Configuration
# ============================================================================
N_SAMPLES_PER_EVAL=(4 4 4 4 1 1)
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

# ============================================================================
# Performance Configuration
# ============================================================================
PERF_ARGS=(
   --tensor-model-parallel-size $TP_SIZE
   --sequence-parallel
   --pipeline-model-parallel-size $PP_SIZE
   --context-parallel-size $CP_SIZE

   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu $MAX_TOKENS_PER_GPU
   --transformer-impl transformer_engine
   --bf16
)

# ============================================================================
# PPO Algorithm Configuration
# ============================================================================
PPO_ARGS=(
   --advantage-estimator ppo
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --kl-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.2
   --num-critic-only-steps 20
   --critic-lr 4e-6
   --critic-num-nodes 1
   --critic-num-gpus-per-node $ACTOR_GPU_COUNT
)


if is_truthy "$ENABLE_CRITIC_TPPO"; then
   PPO_ARGS+=(
      --critic-learn-tppo
      --critic-delta-scale "$CRITIC_DELTA_SCALE"
   )
   if is_truthy "$CRITIC_ZERO_INIT_OUTPUT_LAYER"; then
      PPO_ARGS+=(--critic-zero-init-output-layer)
   fi
   if is_truthy "$CRITIC_TPPO_HEAD_SPACE"; then
      PPO_ARGS+=(--critic-tppo-head-space)
   fi
   if is_truthy "$CRITIC_SEPARATE_V0_HEAD"; then
      PPO_ARGS+=(--critic-separate-v0-head)
      if is_truthy "$CRITIC_SEPARATE_V0_HEAD_LAST_TOKEN_ONLY"; then
         PPO_ARGS+=(--critic-separate-v0-head-last-token-only)
      fi
   fi
fi



# ============================================================================
# Optimizer Configuration
# ============================================================================
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

# ============================================================================
# Wandb Configuration
# ============================================================================
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

# ============================================================================
# SGLang Configuration (Inference Engine)
# ============================================================================
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 2
   --sglang-mem-fraction-static 0.7
   --sglang-cuda-graph-bs 1 2 4 8 $(seq 16 8 256)
)

# ============================================================================
# Miscellaneous Configuration
# ============================================================================
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

# ============================================================================
# Debug Configuration
# ============================================================================
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

# ============================================================================
# Custom Arguments
# ============================================================================
CUSTOM_ARGS=(
   --critic-output-layer-init-mode legacy_v024
   --lambd 1
   --lambd-critic 1
   --log-position-advantage-stats
   --eval-first
)

# ============================================================================
# Launch Ray and Training
# ============================================================================

# launch the master node of ray in container
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus $GPU_COUNT --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=26500

# Build the runtime environment JSON with proper variable substitution
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${SLIME_ROOT}:${MEGATRON_LM_PATH}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\"
  }
}"

mkdir -p "${EXP_RUN_DIR}"
LOG_FILE="${EXP_RUN_DIR}/${EXP_NAME}.log"
echo "Log file will be saved to: ${LOG_FILE}"

export RANK=${NODE_RANK:-0}
export NODE_COUNT=1
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export PROC_PER_NODE=${PROC_PER_NODE:-2}  # 3B uses 2 GPUs for actor (vs 4 for 8B)

if [ -z "$RANK" ]; then
  echo "RANK not set. Defaulting to RANK=0 for single node"
  export RANK=0
fi

echo ""
echo "============================================================================"
echo "Training Configuration Summary"
echo "============================================================================"
echo "Model: llama3.2-3B-Instruct"
echo "Experiment: ${EXP_NAME}"
echo "GPU Configuration:"
echo "  - Total GPUs: ${GPU_COUNT}"
echo "  - Actor GPUs: ${ACTOR_GPU_COUNT}"
echo "  - Tensor Parallel Size: ${TP_SIZE}"
echo "Training Configuration:"
echo "  - Rollout Batch Size: ${ROLLOUT_BATCH_SIZE}"
echo "  - Samples per Prompt: ${N_SAMPLES_PER_PROMPT}"
echo "  - Max Response Length: ${MAX_LEN}"
echo "  - Number of Rollouts: ${NUM_ROLLOUT}"
echo "Paths:"
echo "  - HF Model: ${LLAMA32_3B_HF_PATH}"
echo "  - Megatron Model: ${LLAMA32_3B_MEGATRON_PATH}"
echo "  - Save Dir: ${SAVE_DIR}"
echo "  - Log File: ${LOG_FILE}"
echo "============================================================================"
echo ""

# ========= Submit Ray Job (rank 0 only) =========
if [ "$RANK" == "0" ]; then
   ray job submit --address="http://127.0.0.1:26500" \
      --runtime-env-json="${RUNTIME_ENV_JSON}" \
      -- python3 ${PPO_BASELINE_ROOT}/train.py \
      ${MODEL_ARGS[@]} \
      --actor-num-nodes 1 \
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
