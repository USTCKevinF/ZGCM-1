#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 configs/stages/<stage>.env" >&2
  exit 2
fi

ENV_FILE="$(realpath "$1")"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export ZGCM_REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
source "${ENV_FILE}"
: "${MEGATRON_LM_ROOT:?}"
: "${MEGATRON_VENV:?}"
: "${TORCHRUN_BIN:?}"
: "${TOKENIZER_PATH:?}"
: "${TRAIN_ROOT:?}"
: "${NUM_NODES:?}"
: "${GPUS_PER_NODE:?}"
: "${NODE_RANK:?}"
: "${MASTER_ADDR:?}"
: "${MASTER_PORT:?}"

if [[ ! -f "${MEGATRON_VENV}/bin/activate" ]]; then
  echo "virtual environment not found: ${MEGATRON_VENV}" >&2
  exit 3
fi
source "${MEGATRON_VENV}/bin/activate"
export MEGATRON_HOME="${MEGATRON_LM_ROOT}"
export PYTHONPATH="${TRANSFORMER_ENGINE_HOME:+${TRANSFORMER_ENGINE_HOME}:}${MEGATRON_LM_ROOT}${FLASH_ATTN_V3_EGG:+:${FLASH_ATTN_V3_EGG}}${PYTHONPATH:+:${PYTHONPATH}}"
export NVTE_FRAMEWORK=pytorch
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

for path in "${MEGATRON_LM_ROOT}" "${TOKENIZER_PATH}"; do
  [[ -e "${path}" ]] || { echo "required path not found: ${path}" >&2; exit 3; }
done
if [[ -n "${PRETRAINED_CHECKPOINT:-}" && ! -d "${PRETRAINED_CHECKPOINT}" ]]; then
  echo "pretrained checkpoint not found: ${PRETRAINED_CHECKPOINT}" >&2
  exit 3
fi

if [[ -n "${DATA_ARGS_PATH:-}" ]]; then
  [[ -f "${DATA_ARGS_PATH}" ]] || { echo "data args not found: ${DATA_ARGS_PATH}" >&2; exit 3; }
else
  [[ -f "${REAL_DATA_PREFIX}.idx" && -f "${REAL_DATA_PREFIX}.bin" ]] || {
    echo "indexed dataset not found: ${REAL_DATA_PREFIX}.{idx,bin}" >&2
    exit 3
  }
fi

if [[ "${NO_DATA_SHUFFLE:-0}" == 1 ]]; then
  export MEGATRON_GPT_DATASET_SEQUENTIAL=1
else
  unset MEGATRON_GPT_DATASET_SEQUENTIAL
fi

SAVE_CHECKPOINT_DIR="${SAVE_CHECKPOINT_DIR:-${TRAIN_ROOT}/results/${EXP_NAME}/checkpoints}"
LOAD_CHECKPOINT_DIR="${LOAD_CHECKPOINT_DIR:-${SAVE_CHECKPOINT_DIR}}"
TENSORBOARD_DIR="${TENSORBOARD_DIR:-${TRAIN_ROOT}/results/${EXP_NAME}/tensorboard}"
DATA_CACHE_DIR="${DATA_CACHE_DIR:-${TRAIN_ROOT}/results/${EXP_NAME}/data-cache}"
LOG_DIR="${LOG_DIR:-${TRAIN_ROOT}/logs}"
mkdir -p "${SAVE_CHECKPOINT_DIR}" "${TENSORBOARD_DIR}" "${DATA_CACHE_DIR}" "${LOG_DIR}"

if [[ "${REQUIRE_LOAD_CHECKPOINT:-0}" == 1 ]]; then
  marker="${LOAD_CHECKPOINT_DIR}/latest_checkpointed_iteration.txt"
  [[ -f "${marker}" ]] || { echo "checkpoint marker not found: ${marker}" >&2; exit 4; }
  if [[ -n "${RESUME_ITER:-}" ]]; then
    marker_iter="$(tr -dc '0-9' < "${marker}")"
    [[ "${marker_iter}" == "${RESUME_ITER}" ]] || {
      echo "checkpoint iteration ${marker_iter} != expected ${RESUME_ITER}" >&2
      exit 4
    }
  fi
fi

distributed=(
  --nproc_per_node "${GPUS_PER_NODE}"
  --nnodes "${NUM_NODES}"
  --node_rank "${NODE_RANK}"
  --master_addr "${MASTER_ADDR}"
  --master_port "${MASTER_PORT}"
)

model=(
  --use-mcore-models
  --num-layers "${NUM_LAYERS}"
  --hidden-size "${HIDDEN_SIZE}"
  --ffn-hidden-size "${FFN_HIDDEN_SIZE}"
  --num-attention-heads "${NUM_ATTENTION_HEADS}"
  --group-query-attention
  --num-query-groups "${NUM_QUERY_GROUPS}"
  --kv-channels "${KV_CHANNELS}"
  --seq-length "${SEQ_LENGTH}"
  --max-position-embeddings "${MAX_POSITION_EMBEDDINGS}"
  --position-embedding-type "${POSITION_EMBEDDING_TYPE}"
  --rotary-base "${ROTARY_BASE}"
  --rotary-percent "${ROTARY_PERCENT}"
  --window-size "${WINDOW_SIZE}"
  --window-attn-skip-freq "${WINDOW_ATTN_SKIP_FREQ}"
  --attention-backend "${ATTENTION_BACKEND}"
  --norm-epsilon "${NORM_EPSILON}"
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --swiglu
  --normalization RMSNorm
  --disable-bias-linear
  --untie-embeddings-and-output-weights
  --make-vocab-size-divisible-by 128
  --transformer-impl "${TRANSFORMER_IMPL}"
  --fp8-format "${FP8_FORMAT}"
  --fp8-recipe "${FP8_RECIPE}"
  --fp8-amax-compute-algo "${FP8_AMAX_COMPUTE_ALGO}"
  --fp8-amax-history-len "${FP8_AMAX_HISTORY_LEN}"
)
[[ "${ATTENTION_OUTPUT_GATE:-0}" == 1 ]] && model+=(--attention-output-gate)
[[ "${ATTENTION_OUTPUT_GATE_ONLY_SWA:-0}" == 1 ]] && model+=(--attention-output-gate-only-swa)
[[ "${QK_LAYERNORM:-0}" == 1 ]] && model+=(--qk-layernorm)
[[ "${NO_ROPE_FUSION:-0}" == 1 ]] && model+=(--no-rope-fusion)
[[ "${FP8_PARAM_GATHER:-0}" == 1 ]] && model+=(--fp8-param-gather)

training=(
  --micro-batch-size "${MICRO_BATCH_SIZE}"
  --global-batch-size "${GLOBAL_BATCH_SIZE}"
  --optimizer "${OPTIMIZER}"
  --lr "${LR}"
  --min-lr "${MIN_LR}"
  --lr-decay-style "${LR_DECAY_STYLE}"
  --weight-decay "${WEIGHT_DECAY}"
  --adam-beta1 "${ADAM_BETA1}"
  --adam-beta2 "${ADAM_BETA2}"
  --clip-grad "${CLIP_GRAD}"
  --seed "${SEED}"
  --bf16
  --calculate-per-token-loss
  --muon-momentum "${MUON_MOMENTUM}"
  --muon-scale-mode "${MUON_SCALE_MODE}"
  --muon-fp32-matmul-prec "${MUON_FP32_MATMUL_PREC}"
  --muon-coefficient-type "${MUON_COEFFIC_TYPE}"
  --muon-num-ns-steps "${MUON_NUM_NS_STEPS}"
  --muon-tp-mode "${MUON_TP_MODE}"
  --muon-extra-scale-factor "${MUON_EXTRA_SCALE_FACTOR}"
  --muon-scalar-optimizer "${MUON_SCALAR_OPTIMIZER}"
)

if [[ -n "${TRAIN_ITERS:-}" ]]; then
  training+=(--train-iters "${TRAIN_ITERS}" --lr-warmup-iters "${LR_WARMUP_ITERS:-0}")
else
  training+=(
    --train-samples "${TRAIN_SAMPLES}"
    --lr-warmup-samples "${LR_WARMUP_SAMPLES}"
    --lr-decay-samples "${LR_DECAY_SAMPLES}"
  )
fi
[[ "${USE_DISTRIBUTED_OPTIMIZER:-1}" == 1 ]] && training+=(--use-distributed-optimizer)
[[ "${ENABLE_OVERLAP_GRAD_REDUCE:-1}" == 1 ]] && training+=(--overlap-grad-reduce)
[[ "${ENABLE_OVERLAP_PARAM_GATHER:-0}" == 1 ]] && training+=(--overlap-param-gather)
if [[ "${ENABLE_CROSS_ENTROPY_LOSS_FUSION:-1}" == 1 ]]; then
  training+=(--cross-entropy-loss-fusion)
  [[ -n "${CROSS_ENTROPY_FUSION_IMPL:-}" ]] && training+=(--cross-entropy-fusion-impl "${CROSS_ENTROPY_FUSION_IMPL}")
fi
[[ "${OVERRIDE_OPT_PARAM_SCHEDULER:-0}" == 1 ]] && training+=(--override-opt-param-scheduler)

if [[ "${ENABLE_RECOMPUTE:-0}" == 1 ]]; then
  if [[ "${RECOMPUTE_GRANULARITY:-selective}" == selective ]]; then
    training+=(--recompute-activations --recompute-granularity selective)
  else
    training+=(--recompute-granularity "${RECOMPUTE_GRANULARITY}")
    [[ -n "${RECOMPUTE_METHOD:-}" ]] && training+=(--recompute-method "${RECOMPUTE_METHOD}")
    [[ -n "${RECOMPUTE_NUM_LAYERS:-}" ]] && training+=(--recompute-num-layers "${RECOMPUTE_NUM_LAYERS}")
  fi
fi

parallel=(
  --tensor-model-parallel-size "${TP_SIZE}"
  --pipeline-model-parallel-size "${PP_SIZE}"
  --context-parallel-size "${CP_SIZE}"
)
[[ "${TP_SIZE}" != 1 ]] && parallel+=(--sequence-parallel)
[[ -n "${CP_COMM_TYPE:-}" ]] && parallel+=(--cp-comm-type "${CP_COMM_TYPE}")

data=(
  --vocab-size "${VOCAB_SIZE}"
  --tokenizer-type HuggingFaceTokenizer
  --tokenizer-model "${TOKENIZER_PATH}"
  --data-cache-path "${DATA_CACHE_DIR}"
  --split "${DATA_SPLIT:-100,0,0}"
  --no-create-attention-mask-in-dataloader
  --num-workers "${DATA_NUM_WORKERS:-4}"
  --num-dataset-builder-threads "${NUM_DATASET_BUILDER_THREADS:-8}"
)
if [[ -n "${DATA_ARGS_PATH:-}" ]]; then
  data+=(--data-args-path "${DATA_ARGS_PATH}")
else
  data+=(--data-path "${REAL_DATA_PREFIX}")
fi

logging=(
  --log-interval "${LOG_INTERVAL:-10}"
  --save-interval "${SAVE_INTERVAL:-1000}"
  --eval-interval "${EVAL_INTERVAL:-100000}"
  --eval-iters "${EVAL_ITERS:-0}"
  --log-throughput
  --ckpt-format torch_dist
  --distributed-timeout-minutes 120
  --tensorboard-dir "${TENSORBOARD_DIR}"
  --save "${SAVE_CHECKPOINT_DIR}"
  --load "${LOAD_CHECKPOINT_DIR}"
  --rerun-mode "${RERUN_MODE:-disabled}"
)
[[ -n "${PRETRAINED_CHECKPOINT:-}" ]] && logging+=(--pretrained-checkpoint "${PRETRAINED_CHECKPOINT}")

cd "${MEGATRON_LM_ROOT}"
run_tag="${EXP_NAME}_$(date -u +%Y%m%d_%H%M%S)"
echo "run_tag=${run_tag} env=${ENV_FILE} save=${SAVE_CHECKPOINT_DIR} load=${LOAD_CHECKPOINT_DIR}"
exec "${TORCHRUN_BIN}" "${distributed[@]}" "${TRAIN_PROGRAM}" \
  "${model[@]}" "${training[@]}" "${parallel[@]}" "${data[@]}" "${logging[@]}" \
  2>&1 | tee -a "${LOG_DIR}/${run_tag}.log"
