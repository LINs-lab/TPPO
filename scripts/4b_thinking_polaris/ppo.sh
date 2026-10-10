#!/bin/bash


if [[ "${ALLOW_GLOBAL_RUNTIME_CLEANUP:-0}" == "1" ]]; then
    pkill -9 sglang || true
    ray stop --force || true
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
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"




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


METHOD="${METHOD:-PPO}"
EXP_NAME="${EXP_NAME:-4B-Thinking-PPO-Polaris-n1-bs128-w50}"


source "${MODEL_CONFIG_PATH:-${PPO_BASELINE_ROOT}/scripts/models/qwen3-4B.sh}"
MODEL_ARGS+=(
   --max-position-embeddings 32768
   --seq-length 32768
)
WANDB_RUN_NAME="${EXP_NAME}_$(date +%Y%m%d_%H%M%S)"

SAVE_DIR="${SAVE_DIR:-${TPPO_ROOT}/outputs/checkpoints/${METHOD}/${EXP_NAME}/}"
LOG_DIR="${LOG_DIR:-${TPPO_ROOT}/outputs/logs}"
EXP_RUN_DIR="${LOG_DIR}/${METHOD}/${EXP_NAME}"
mkdir -p "${EXP_RUN_DIR}" "${EXP_RUN_DIR}/wandb"
CRITIC_SAVE_DIR="${SAVE_DIR}/critic"
CRITIC2_SAVE_DIR="${SAVE_DIR}/critic2"
   mkdir -p "${SAVE_DIR}" "${CRITIC_SAVE_DIR}" "${CRITIC2_SAVE_DIR}"
GPU_COUNT=8
ACTOR_GPU_COUNT=4
TP_SIZE=$ACTOR_GPU_COUNT
PP_SIZE=1
CP_SIZE=1
EP_SIZE=1
ETP_SIZE=1
MAX_LEN="${MAX_LEN:-$((1024 * 8))}"
MAX_TOKENS_PER_GPU=$((($MAX_LEN / $CP_SIZE) + 1024))
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-128}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-1}"
NUM_STEPS_PER_ROLLOUT="${NUM_STEPS_PER_ROLLOUT:-2}"
NUM_ROLLOUT="${NUM_ROLLOUT:-520}"
START_ROLLOUT_ARG=()
# Check if resume checkpoint exists
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
# ========================================================================

# Set checkpoint arguments based on resume status
if [ "$USE_RESUME_CHECKPOINT" = true ]; then
    CKPT_ARGS=(
      #  --load ${RESUME_CHECKPOINT_DIR}
      #  --start-rollout-id ${RESUME_ROLLOUT_ID}
      #  --save ${RESUME_CHECKPOINT_DIR}
       
      #  $(if [ "$USE_RESUME_CRITIC_CHECKPOINT" = true ]; then echo "--critic-load ${RESUME_CRITIC_CHECKPOINT_DIR}"; fi)
      #  --critic-save ${CRITIC_SAVE_DIR}
       
      #  # Critic2 configuration (second critic model)
      #  --use-critic2
      #  $(if [ "$USE_RESUME_CRITIC2_CHECKPOINT" = true ]; then echo "--critic2-load ${RESUME_CRITIC2_CHECKPOINT_DIR}"; fi)
      #  --critic2-save ${CRITIC2_SAVE_DIR}
       
      #  --save-interval 500
    )
else
    CKPT_ARGS=(
       --megatron-to-hf-mode bridge
       --hf-checkpoint "${HF_MODEL_PATH:-/path/to/models/Qwen3-4B}"
       --ref-load "${MEGATRON_MODEL_PATH:-/path/to/models/Qwen3-4B-megatron}"
       --load "${MEGATRON_MODEL_PATH:-/path/to/models/Qwen3-4B-megatron}"
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
   --prompt-data ${TPPO_ROOT}/data/polaris/qwen3-4b-s1-polaris-p1-format.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --apply-chat-template-kwargs '{"enable_thinking":false}'
   --rollout-shuffle
   --rm-type p1
   --rm-timeout-seconds 600
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
   --use-tis
   --balance-data
   # --dynamic-sampling-filter-path slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std
)
N_SAMPLES_PER_EVAL=(4 4 4 4 1 1)
# N_SAMPLES_PER_EVAL=(32 32 32 32 1 1)
IMO_PATH=/path/to/test_jsonl
EVAL_ARGS=(
   --eval-interval "${EVAL_INTERVAL:-10}"
   --skip-eval-during-critic-warmup
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
   # --expert-model-parallel-size $EP_SIZE
   # --expert-tensor-parallel-size $ETP_SIZE

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
   --num-critic-only-steps "${NUM_CRITIC_ONLY_STEPS:-50}"
   # --num-critic-only-steps 1
   # --normalize-advantages
   --critic-lr 4e-6
   --critic-num-nodes 1
   --critic-num-gpus-per-node $ACTOR_GPU_COUNT

   # Critic2 configuration (same settings as critic1)
   # --critic2-num-nodes 1
   # --critic2-num-gpus-per-node 2
   # --critic2-lr 4e-6
)

if [[ "${CRITIC_ZERO_INIT_OUTPUT_LAYER:-0}" == "1" ]]; then
   PPO_ARGS+=(--critic-zero-init-output-layer)
fi

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
   --rollout-num-gpus-per-engine 2
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
   --critic-output-layer-init-mode "${CRITIC_OUTPUT_LAYER_INIT_MODE:-legacy_v024}"
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
export PROC_PER_NODE=${PROC_PER_NODE:-4}

if [ -z "$RANK" ]; then
  echo "RANK not set. Defaulting to RANK=0 for single node"
  export RANK=0
fi


# ========= Submit Ray Job (rank 0 only) =========
if [ "$RANK" == "0" ]; then
   ray job submit --address="http://127.0.0.1:26500" \
      --runtime-env-json="${RUNTIME_ENV_JSON}" \
      -- python3 "${PPO_BASELINE_ROOT}/train.py" \
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
