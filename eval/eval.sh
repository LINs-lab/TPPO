#!/bin/bash

# Evaluate a supported HF checkpoint on the copied JSONL benchmarks.
# Settings are aligned with the 8B training val setup:
#   GPU_COUNT=8, ACTOR_GPU_COUNT=8, CP_SIZE=1,
#   TP_SIZE is set per checkpoint to match the training checkpoint TP,
#   rollout-num-gpus-per-engine is set to the same TP so engines = 8 / TP,
#   rollout max response len=8k, eval max response len=24k,
#   eval top_p=0.95, eval temperature=1.0.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TPPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
SLIME_ROOT="${TPPO_ROOT}/slime"
MEGATRON_LM_PATH="${MEGATRON_LM_PATH:-/path/to/Megatron-LM}"
export TPPO_ROOT SLIME_ROOT MEGATRON_LM_PATH
export PYTHONPATH="${SLIME_ROOT}:${MEGATRON_LM_PATH}:${PYTHONPATH:-}"
REPO_DIR="${SLIME_ROOT}"
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    cat <<'EOF'
Usage: bash eval/eval.sh MODEL
Models: 3b_llama, 4b_polaris, 4b_thinking_polaris, 8b_polaris, countdown
Set HF_CKPT_ROOT to the Hugging Face checkpoint being evaluated.
Set BASE_MEGATRON_CKPT and MEGATRON_LM_PATH to your local model/runtime paths.
Optional: EVAL_DATASETS=all|math|ood|aime24|aime25|aime26|amc23|math500|
                        minervamath|olympic|arc_c|agi_eval_en|bigbenchhard
          GPU_COUNT=4 TP_SIZE=2
Default: all (the seven math and three OOD benchmarks).
All evaluation datasets default to 8 responses per prompt.
EOF
    exit 0
fi
MODEL="${1:-${MODEL:-8b_polaris}}"
case "${MODEL}" in
    3b_llama|3b)
        MODEL_CONFIG="${MODEL_CONFIG:-${SLIME_ROOT}/scripts/models/llama3.2-3B-Instruct.sh}"
        BASE_MEGATRON_CKPT="${BASE_MEGATRON_CKPT:-/path/to/models/llama3.2-3B-Instruct-megatron}"
        GPU_COUNT="${GPU_COUNT:-4}"; TP_SIZE="${TP_SIZE:-2}"
        ;;
    4b_polaris|4b)
        MODEL_CONFIG="${MODEL_CONFIG:-${SLIME_ROOT}/scripts/models/qwen3-4B.sh}"
        BASE_MEGATRON_CKPT="${BASE_MEGATRON_CKPT:-/path/to/models/Qwen3-4B-Base-megatron}"
        ;;
    4b_thinking_polaris|4b_thinking)
        MODEL_CONFIG="${MODEL_CONFIG:-${SLIME_ROOT}/scripts/models/qwen3-4B.sh}"
        BASE_MEGATRON_CKPT="${BASE_MEGATRON_CKPT:-/path/to/models/Qwen3-4B-megatron}"
        if [[ -z "${APPLY_CHAT_TEMPLATE_KWARGS:-}" ]]; then
            APPLY_CHAT_TEMPLATE_KWARGS='{"enable_thinking":false}'
        fi
        ;;
    8b_polaris|8b)
        MODEL_CONFIG="${MODEL_CONFIG:-${TPPO_ROOT}/scripts/models/qwen3-8B-Base.sh}"
        BASE_MEGATRON_CKPT="${BASE_MEGATRON_CKPT:-/path/to/models/Qwen3-8B-Base-megatron}"
        ;;
    countdown)
        MODEL_CONFIG="${MODEL_CONFIG:-${SLIME_ROOT}/scripts/models/qwen2.5-1.5B.sh}"
        BASE_MEGATRON_CKPT="${BASE_MEGATRON_CKPT:-/path/to/models/Qwen2.5-Math-1.5B-megatron}"
        GPU_COUNT="${GPU_COUNT:-4}"; TP_SIZE="${TP_SIZE:-1}"
        MAX_LEN="${MAX_LEN:-4096}"
        EVAL_MAX_RESPONSE_LEN="${EVAL_MAX_RESPONSE_LEN:-8192}"
        EXTRA_MODEL_ARGS="${EXTRA_MODEL_ARGS:---max-position-embeddings 16384 --seq-length 16384}"
        export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
        ;;
    *) echo "Unsupported model: ${MODEL}" >&2; exit 2 ;;
esac
cd "${REPO_DIR}"
export PYTHONBUFFERED=16 PYTHONNOUSERSITE=1
RAY_BIN="${RAY_BIN:-ray}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
if [[ "${ALLOW_GLOBAL_RUNTIME_CLEANUP:-0}" == "1" ]]; then
    pkill -9 sglang || true
    "${RAY_BIN}" stop --force || true
fi
NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l || true)
HAS_NVLINK=0
if (( NVLINK_COUNT > 0 )); then HAS_NVLINK=1; fi
export N_SAMPLES_PER_EVAL_DEFAULT="${N_SAMPLES_PER_EVAL_DEFAULT:-8}"

HF_CKPT_ROOT="${HF_CKPT_ROOT:?Set HF_CKPT_ROOT to an existing HF checkpoint directory.}"
MODEL_TAG="${MODEL_TAG:-$(basename "$(dirname "${HF_CKPT_ROOT}")")-$(basename "${HF_CKPT_ROOT}")}"
METHOD="${METHOD:-HLG-Eval}"
EXP_NAME="${EXP_NAME:-${MODEL_TAG}-valid-ood-mc-eval8g}"

if [ ! -f "${HF_CKPT_ROOT}/config.json" ]; then
    echo "Missing HF checkpoint config: ${HF_CKPT_ROOT}/config.json" >&2
    exit 1
fi

DATA_DIR="${DATA_DIR:-${TPPO_ROOT}/data/eval}"
ARC_C_DATA="${ARC_C_DATA:-${DATA_DIR}/arc_c_prompted.jsonl}"
AGI_EVAL_EN_DATA="${AGI_EVAL_EN_DATA:-${DATA_DIR}/agi_eval_en_prompted.jsonl}"
BIGBENCHHARD_DATA="${BIGBENCHHARD_DATA:-${DATA_DIR}/bigbenchhard_prompted.jsonl}"
AIME24_DATA="${AIME24_DATA:-${DATA_DIR}/aime_2024.jsonl}"
AIME25_DATA="${AIME25_DATA:-${DATA_DIR}/aime_2025.jsonl}"
AIME26_DATA="${AIME26_DATA:-${DATA_DIR}/aime_2026.jsonl}"
AMC23_DATA="${AMC23_DATA:-${DATA_DIR}/amc23.jsonl}"
MATH500_DATA="${MATH500_DATA:-${DATA_DIR}/math500.jsonl}"
MINERVAMATH_DATA="${MINERVAMATH_DATA:-${DATA_DIR}/minervamath.jsonl}"
OLYMPIC_DATA="${OLYMPIC_DATA:-${DATA_DIR}/olympic.jsonl}"

source "${MODEL_CONFIG}"
if [ -n "${EXTRA_MODEL_ARGS:-}" ]; then
   read -r -a EXTRA_MODEL_ARGS_ARRAY <<< "${EXTRA_MODEL_ARGS}"
   MODEL_ARGS+=("${EXTRA_MODEL_ARGS_ARRAY[@]}")
fi

BASE_MEGATRON_CKPT="${BASE_MEGATRON_CKPT:-/path/to/models/Qwen3-8B-Base-megatron}"

LOG_DIR="${LOG_DIR:-${TPPO_ROOT}/outputs/logs}"
EXP_RUN_DIR="${LOG_DIR}/${METHOD}/${EXP_NAME}"
DEBUG_DIR="${EXP_RUN_DIR}/debug_data"
ANALYSIS_DIR="${DEBUG_DIR}/analysis"
mkdir -p "${EXP_RUN_DIR}" "${EXP_RUN_DIR}/wandb" "${ANALYSIS_DIR}"

# slime_B treats a path whose basename is iter_XXXXXXX as a Megatron checkpoint.
# Use a stable non-iter symlink so the HF loader branch is selected.
HF_LOAD_PATH="${EXP_RUN_DIR}/hf_actor"
ln -sfn "${HF_CKPT_ROOT}" "${HF_LOAD_PATH}"
echo "HF checkpoint source: ${HF_CKPT_ROOT}"
echo "HF checkpoint load path: ${HF_LOAD_PATH}"

GPU_COUNT="${GPU_COUNT:-8}"
NUM_GPUS_PER_NODE="${NUM_GPUS_PER_NODE:-${GPU_COUNT}}"
ACTOR_GPU_COUNT="${ACTOR_GPU_COUNT:-${GPU_COUNT}}"
TP_SIZE="${TP_SIZE:-4}"
PP_SIZE="${PP_SIZE:-1}"
CP_SIZE="${CP_SIZE:-1}"
ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-${ACTOR_GPU_COUNT}}"
ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:-${TP_SIZE}}"

if [ $((ACTOR_GPU_COUNT % TP_SIZE)) -ne 0 ]; then
   echo "ACTOR_GPU_COUNT=${ACTOR_GPU_COUNT} must be divisible by TP_SIZE=${TP_SIZE}." >&2
   exit 1
fi
if [ "${ROLLOUT_NUM_GPUS_PER_ENGINE}" -ne "${TP_SIZE}" ]; then
   echo "ROLLOUT_NUM_GPUS_PER_ENGINE=${ROLLOUT_NUM_GPUS_PER_ENGINE} must match TP_SIZE=${TP_SIZE} for training-TP-aligned eval." >&2
   exit 1
fi
if [ $((ROLLOUT_NUM_GPUS % ROLLOUT_NUM_GPUS_PER_ENGINE)) -ne 0 ]; then
   echo "ROLLOUT_NUM_GPUS=${ROLLOUT_NUM_GPUS} must be divisible by ROLLOUT_NUM_GPUS_PER_ENGINE=${ROLLOUT_NUM_GPUS_PER_ENGINE}." >&2
   exit 1
fi
ROLLOUT_ENGINE_COUNT=$((ROLLOUT_NUM_GPUS / ROLLOUT_NUM_GPUS_PER_ENGINE))

MAX_LEN="${MAX_LEN:-$((1024 * 8))}"
EVAL_MAX_RESPONSE_LEN="${EVAL_MAX_RESPONSE_LEN:-$((MAX_LEN * 3))}"
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-$(((MAX_LEN / CP_SIZE) + 1024))}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-64}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-8}"
EFFECTIVE_RM_URL="${RM_URL:-http://127.0.0.1:8001/}"
EFFECTIVE_XVERIFY_MODEL_NAME="${XVERIFY_MODEL_NAME:-gpt-oss-120b}"

if [ "${USE_XVERIFY:-0}" = "1" ]; then
   if [ -z "${RM_URL:-}" ]; then
      echo "USE_XVERIFY=1 requires RM_URL to point to an OpenAI-compatible judge endpoint, e.g. http://HOST:34882/v1." >&2
      exit 1
   fi
   case "${RM_URL%/}" in
      */v1) ;;
      *)
         echo "USE_XVERIFY=1 expects RM_URL to end with /v1, got: ${RM_URL}" >&2
         exit 1
         ;;
   esac
fi

case "${EVAL_DATASETS:-all}" in
   all)
      EVAL_DATASET_NAMES=(aime24 aime25 aime26 amc23 math500 minervamath olympic arc_c agi_eval_en bigbenchhard)
      ;;
   math|math_bench)
      EVAL_DATASET_NAMES=(aime24 aime25 aime26 amc23 math500 minervamath olympic)
      ;;
   ood)
      EVAL_DATASET_NAMES=(arc_c agi_eval_en bigbenchhard)
      ;;
   aime24|aime25|aime26|amc23|math500|minervamath|olympic|arc_c|agi_eval_en|bigbenchhard)
      EVAL_DATASET_NAMES=("${EVAL_DATASETS}")
      ;;
   *)
      echo "Unsupported EVAL_DATASETS=${EVAL_DATASETS}. Use all, math, ood, aime24, aime25, aime26, amc23, math500, minervamath, olympic, arc_c, agi_eval_en, or bigbenchhard." >&2
      exit 1
      ;;
esac

declare -A EVAL_DATASET_PATHS=(
   [aime24]="${AIME24_DATA}"
   [aime25]="${AIME25_DATA}"
   [aime26]="${AIME26_DATA}"
   [amc23]="${AMC23_DATA}"
   [math500]="${MATH500_DATA}"
   [minervamath]="${MINERVAMATH_DATA}"
   [olympic]="${OLYMPIC_DATA}"
   [arc_c]="${ARC_C_DATA}"
   [agi_eval_en]="${AGI_EVAL_EN_DATA}"
   [bigbenchhard]="${BIGBENCHHARD_DATA}"
)
declare -A EVAL_DATASET_SAMPLES=(
   [aime24]="${AIME24_N_SAMPLES_PER_EVAL:-${N_SAMPLES_PER_EVAL_DEFAULT}}"
   [aime25]="${AIME25_N_SAMPLES_PER_EVAL:-${N_SAMPLES_PER_EVAL_DEFAULT}}"
   [aime26]="${AIME26_N_SAMPLES_PER_EVAL:-${N_SAMPLES_PER_EVAL_DEFAULT}}"
   [amc23]="${AMC23_N_SAMPLES_PER_EVAL:-${N_SAMPLES_PER_EVAL_DEFAULT}}"
   [math500]="${MATH500_N_SAMPLES_PER_EVAL:-${N_SAMPLES_PER_EVAL_DEFAULT}}"
   [minervamath]="${MINERVAMATH_N_SAMPLES_PER_EVAL:-${N_SAMPLES_PER_EVAL_DEFAULT}}"
   [olympic]="${OLYMPIC_N_SAMPLES_PER_EVAL:-${N_SAMPLES_PER_EVAL_DEFAULT}}"
   [arc_c]="${ARC_C_N_SAMPLES_PER_EVAL:-${N_SAMPLES_PER_EVAL_DEFAULT}}"
   [agi_eval_en]="${AGI_EVAL_EN_N_SAMPLES_PER_EVAL:-${N_SAMPLES_PER_EVAL_DEFAULT}}"
   [bigbenchhard]="${BIGBENCHHARD_N_SAMPLES_PER_EVAL:-${N_SAMPLES_PER_EVAL_DEFAULT}}"
)
EVAL_PROMPT_DATA=()
N_SAMPLES_PER_EVAL=()
for dataset in "${EVAL_DATASET_NAMES[@]}"; do
   EVAL_PROMPT_DATA+=("${dataset}" "${EVAL_DATASET_PATHS[$dataset]}")
   N_SAMPLES_PER_EVAL+=("${EVAL_DATASET_SAMPLES[$dataset]}")
done

if [ -n "${N_SAMPLES_PER_EVAL_OVERRIDE:-}" ]; then
   read -r -a N_SAMPLES_PER_EVAL <<< "${N_SAMPLES_PER_EVAL_OVERRIDE}"
fi

for ((i = 1; i < ${#EVAL_PROMPT_DATA[@]}; i += 2)); do
   data_path="${EVAL_PROMPT_DATA[$i]}"
   if [ ! -f "${data_path}" ]; then
      echo "Missing eval data: ${data_path}" >&2
      echo "Set DATA_DIR or the corresponding dataset path to the copied data." >&2
      exit 1
   fi
done

CKPT_ARGS=(
   --megatron-to-hf-mode bridge
   --hf-checkpoint "${HF_LOAD_PATH}"
   --ref-load "${BASE_MEGATRON_CKPT}"
   --load "${HF_LOAD_PATH}"
   --no-load-optim
   --no-load-rng
)

ROLLOUT_ARGS=(
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rm-type "${RM_TYPE:-p1}"
   --rm-url "${EFFECTIVE_RM_URL}"
   --reward-key score

   --num-rollout 0
   --rollout-num-gpus $ROLLOUT_NUM_GPUS
   --rollout-batch-size $ROLLOUT_BATCH_SIZE
   --n-samples-per-prompt $N_SAMPLES_PER_PROMPT
   --rollout-max-response-len $MAX_LEN
   --rollout-temperature 1.0
)

if [ -n "${APPLY_CHAT_TEMPLATE_KWARGS:-}" ]; then
   ROLLOUT_ARGS+=(--apply-chat-template-kwargs "${APPLY_CHAT_TEMPLATE_KWARGS}")
fi

if [ "${USE_XVERIFY:-0}" = "1" ]; then
   ROLLOUT_ARGS+=("--train-use-xverify")
fi

EVAL_ARGS=(
   --eval-interval 1
   --eval-prompt-data "${EVAL_PROMPT_DATA[@]}"
   --n-samples-per-eval-prompt "${N_SAMPLES_PER_EVAL[@]}"
   --eval-max-response-len "${EVAL_MAX_RESPONSE_LEN}"
   --eval-top-p 0.95
   --eval-temperature 1.0
   --log-passrate
)

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

# Eval generation only needs the actor. Keep GRPO-style args to avoid critic init.
PPO_ARGS=(
   --advantage-estimator grpo
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --kl-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.2
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --lr-decay-iters 1
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98

   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
)

export WANDB_DIR="${EXP_RUN_DIR}/wandb"
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_MODE="${WANDB_MODE:-offline}"

WANDB_ARGS=(
   --use-wandb
   --wandb-project "${METHOD}"
   --wandb-group "${EXP_NAME}_$(date +%Y%m%d_%H%M%S)"
   --disable-wandb-random-suffix
   --wandb-dir "${WANDB_DIR}"
   --wandb-mode "${WANDB_MODE}"
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine $ROLLOUT_NUM_GPUS_PER_ENGINE
   --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC:-0.7}"
   --sglang-cuda-graph-bs 1 2 4 8 $(seq 16 8 256)
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

DEBUG_ARGS=(
   --dump-details "${ANALYSIS_DIR}"
)

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
"${RAY_BIN}" start --head --node-ip-address ${MASTER_ADDR} --num-gpus $GPU_COUNT --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=26500

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${REPO_DIR}:${MEGATRON_LM_PATH}/\",
    \"PYTHONNOUSERSITE\": \"1\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"XVERIFY_BASE_URL\": \"${EFFECTIVE_RM_URL}\",
    \"XVERIFY_MODEL_NAME\": \"${EFFECTIVE_XVERIFY_MODEL_NAME}\",
    \"XVERIFY_API_KEY\": \"${XVERIFY_API_KEY:-None}\",
    \"XVERIFY_MAX_TOKENS\": \"${XVERIFY_MAX_TOKENS:-4096}\",
    \"XVERIFY_TEMPERATURE\": \"${XVERIFY_TEMPERATURE:-0.1}\"
  }
}"

LOG_FILE="${EXP_RUN_DIR}/${EXP_NAME}.log"
echo "Log file will be saved to: ${LOG_FILE}"
echo "Eval debug data will be saved under: ${ANALYSIS_DIR}/rollout_data"
echo "EVAL_PROMPT_DATA=${EVAL_PROMPT_DATA[*]}"
echo "N_SAMPLES_PER_EVAL=${N_SAMPLES_PER_EVAL[*]}"
echo "USE_XVERIFY=${USE_XVERIFY:-0} RM_URL=${EFFECTIVE_RM_URL} XVERIFY_MODEL_NAME=${EFFECTIVE_XVERIFY_MODEL_NAME}"
echo "GPU layout: GPU_COUNT=${GPU_COUNT}, ACTOR_GPU_COUNT=${ACTOR_GPU_COUNT}, TP_SIZE=${TP_SIZE}, ROLLOUT_NUM_GPUS=${ROLLOUT_NUM_GPUS}, ROLLOUT_NUM_GPUS_PER_ENGINE=${ROLLOUT_NUM_GPUS_PER_ENGINE}, ROLLOUT_ENGINE_COUNT=${ROLLOUT_ENGINE_COUNT}"

export RANK=${NODE_RANK:-0}
export NODE_COUNT=1
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export PROC_PER_NODE=${PROC_PER_NODE:-8}

if [ "$RANK" == "0" ]; then
   "${RAY_BIN}" job submit --address="http://127.0.0.1:26500" \
      --runtime-env-json="${RUNTIME_ENV_JSON}" \
      -- "${PYTHON_BIN}" "${REPO_DIR}/train.py" \
      "${MODEL_ARGS[@]}" \
      --actor-num-nodes 1 \
      --actor-num-gpus-per-node $ACTOR_GPU_COUNT \
      --num-gpus-per-node $NUM_GPUS_PER_NODE \
      --colocate \
      "${CKPT_ARGS[@]}" \
      "${ROLLOUT_ARGS[@]}" \
      "${OPTIMIZER_ARGS[@]}" \
      "${PPO_ARGS[@]}" \
      "${WANDB_ARGS[@]}" \
      "${PERF_ARGS[@]}" \
      "${EVAL_ARGS[@]}" \
      "${SGLANG_ARGS[@]}" \
      "${DEBUG_ARGS[@]}" \
      "${MISC_ARGS[@]}" 2>&1 | tee -a "${LOG_FILE}"
fi
