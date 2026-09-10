#!/usr/bin/env bash
set -Eeuo pipefail

ulimit -n 1048576 || true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
VARIANT="${VARIANT:?VARIANT is required}"
PROJECT_ROOT="${PROJECT_ROOT:?PROJECT_ROOT is required}"
RUN_NAME="${RUN_NAME:-${VARIANT}}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_ROOT}/megatron_runs/${RUN_NAME}}"
SOURCE_CKPT_ROOT="${SOURCE_CKPT_ROOT:?SOURCE_CKPT_ROOT is required}"
SOURCE_CKPT_ITER="${SOURCE_CKPT_ITER:?SOURCE_CKPT_ITER is required}"
CKPT_LOAD_ITER="${CKPT_LOAD_ITER:?CKPT_LOAD_ITER is required}"
CKPT_LOAD_DIR_NAME="$(printf 'iter_%07d' "${CKPT_LOAD_ITER}")"
DATA_ROOT="${DATA_ROOT:?DATA_ROOT is required}"
QCORE_ROOT="${QCORE_ROOT:-${DATA_ROOT}}"
THINK_ROOT="${THINK_ROOT:-${DATA_ROOT}}"
TOOL_ROOT="${TOOL_ROOT:-${DATA_ROOT}}"
TOKENIZER="${TOKENIZER:?TOKENIZER is required}"
SOURCE_DATA_ARGS="${SOURCE_DATA_ARGS:?SOURCE_DATA_ARGS is required}"
INDEXED_DONE="${INDEXED_DONE:?INDEXED_DONE is required}"

LOAD_PARENT="${RUN_ROOT}/load_ckpt"
TB_DIR="${RUN_ROOT}/tensorboard"
CACHE_DIR="${RUN_ROOT}/data-cache"
DATA_DIR="${RUN_ROOT}/data"
LOG_DIR="${RUN_ROOT}/logs"
CKPT_DIR="${RUN_ROOT}/checkpoints"
STATUS_DIR="${RUN_ROOT}/status"
CONFIG_DIR="${RUN_ROOT}/configs"

NODE_RANK_EARLY="${NODE_RANK:-${RANK:-0}}"
if [[ "${NODE_RANK_EARLY}" == "0" ]]; then
  mkdir -p "${LOAD_PARENT}" "${TB_DIR}" "${CACHE_DIR}" "${DATA_DIR}" "${LOG_DIR}" "${CKPT_DIR}" "${STATUS_DIR}" "${CONFIG_DIR}" "${PROJECT_ROOT}/status"
  rm -f "${STATUS_DIR}/train.done" "${STATUS_DIR}/train.failed" "${PROJECT_ROOT}/status/train.done" "${PROJECT_ROOT}/status/train.failed"
  ln -sfn "${SOURCE_CKPT_ITER}" "${LOAD_PARENT}/${CKPT_LOAD_DIR_NAME}"
  printf "%s\n" "${CKPT_LOAD_ITER}" > "${LOAD_PARENT}/latest_checkpointed_iteration.txt"
  tmp_data_args="${DATA_DIR}/per_split_data_args.json.tmp.$$"
  cp "${SOURCE_DATA_ARGS}" "${tmp_data_args}"
  mv -f "${tmp_data_args}" "${DATA_DIR}/per_split_data_args.json"
  cat > "${CONFIG_DIR}/run_manifest.json.tmp.$$" <<EOF
{
  "project_root": "${PROJECT_ROOT}",
  "run_name": "${RUN_NAME}",
  "run_root": "${RUN_ROOT}",
  "source_ckpt_root": "${SOURCE_CKPT_ROOT}",
  "source_ckpt_iter": "${SOURCE_CKPT_ITER}",
  "ckpt_load_iter": "${CKPT_LOAD_ITER}",
  "source_data_args": "${SOURCE_DATA_ARGS}",
  "qcore_root": "${QCORE_ROOT}",
  "think_root": "${THINK_ROOT}",
  "tool_root": "${TOOL_ROOT}",
  "mix_train_weight_qcore": ${MIX_TRAIN_WEIGHT_QCORE},
  "mix_train_weight_think": ${MIX_TRAIN_WEIGHT_THINK},
  "mix_train_weight_tool": ${MIX_TRAIN_WEIGHT_TOOL},
  "tokenizer": "${TOKENIZER}",
  "seq_length": ${SEQ_LENGTH},
  "qcore_train_records": ${QCORE_TRAIN_RECORDS},
  "qcore_train_tokens": ${QCORE_TRAIN_TOKENS},
  "think_train_records": ${THINK_TRAIN_RECORDS},
  "think_train_tokens": ${THINK_TRAIN_TOKENS},
  "tool_train_records": ${TOOL_TRAIN_RECORDS},
  "tool_raw_records": ${TOOL_RAW_RECORDS},
  "tool_train_tokens": ${TOOL_TRAIN_TOKENS},
  "tool_labeled_tokens": ${TOOL_LABELED_TOKENS},
  "epoch_iters": ${EPOCH_ITERS},
  "train_iters": ${TRAIN_ITERS},
  "exit_interval": null,
  "lr": "${LR}",
  "min_lr": "${MIN_LR}",
  "lr_decay_style": "${LR_DECAY_STYLE}",
  "lr_warmup_iters": ${LR_WARMUP_ITERS},
  "optimizer": "${OPTIMIZER}",
  "weight_decay": "${WEIGHT_DECAY}",
  "global_batch_size": ${GLOBAL_BATCH_SIZE},
  "precision": "bf16_with_te_fp8_hybrid_delayed_${ATTENTION_MODE}_${SEQ_LENGTH}_${RECOMPUTE_MODE}",
  "fp8_format": "hybrid",
  "fp8_recipe": "delayed",
  "fp8_amax_compute_algo": "max",
  "fp8_amax_history_len": 1024,
  "fp8_param_gather": true,
  "total_train_records": ${MIX_TOTAL_TRAIN_RECORDS},
  "save_interval": ${SAVE_INTERVAL},
  "worker_num": ${WORKER_NUM},
  "gpus_per_worker": ${GPUS_PER_WORKER},
  "checkpoint_save_strategy": "torch_dist_sync_mcore_wrapper",
  "swanlab_project": "${SWANLAB_PROJECT:-zgcm-s4-sft}",
  "created_at": "$(date -Is)"
}
EOF
  mv -f "${CONFIG_DIR}/run_manifest.json.tmp.$$" "${CONFIG_DIR}/run_manifest.json"
  touch "${RUN_ROOT}/rank0_setup.done"
else
  for _i in $(seq 1 900); do
    if [[ -f "${RUN_ROOT}/rank0_setup.done" && -f "${DATA_DIR}/per_split_data_args.json" && -f "${LOAD_PARENT}/latest_checkpointed_iteration.txt" ]]; then
      break
    fi
    sleep 1
  done
  if [[ ! -f "${RUN_ROOT}/rank0_setup.done" ]]; then
    echo "rank0 setup did not finish in time" >&2
    exit 1
  fi
  mkdir -p "${TB_DIR}" "${CACHE_DIR}" "${LOG_DIR}" "${CKPT_DIR}" "${STATUS_DIR}" "${CONFIG_DIR}"
fi

TELEMETRY_PID=""
SWANLAB_PID=""
finalize() {
  rc=$?
  if [[ -n "${TELEMETRY_PID}" ]]; then
    kill "${TELEMETRY_PID}" 2>/dev/null || true
  fi
  if [[ -n "${SWANLAB_PID}" ]]; then
    kill "${SWANLAB_PID}" 2>/dev/null || true
  fi
  if [[ ${rc} -ne 0 && "${NODE_RANK_EARLY}" == "0" ]]; then
    touch "${STATUS_DIR}/train.failed"
    touch "${PROJECT_ROOT}/status/train.failed"
  fi
  exit "${rc}"
}
trap finalize EXIT

for required in \
  "${REPO}" \
  "${SOURCE_CKPT_ITER}" \
  "${SOURCE_CKPT_ITER}/.metadata" \
  "${SOURCE_CKPT_ITER}/common.pt" \
  "${SOURCE_DATA_ARGS}" \
  "${INDEXED_DONE}" \
  "${TOKENIZER}/tokenizer_config.json"; do
  if [[ ! -e "${required}" ]]; then
    echo "missing required path: ${required}" >&2
    exit 2
  fi
done

if [[ "$(tr -d '[:space:]' < "${LOAD_PARENT}/latest_checkpointed_iteration.txt")" != "${CKPT_LOAD_ITER}" ]]; then
  echo "unexpected static load checkpoint iteration: $(cat "${LOAD_PARENT}/latest_checkpointed_iteration.txt") expected ${CKPT_LOAD_ITER}" >&2
  exit 2
fi
if [[ ! -L "${LOAD_PARENT}/${CKPT_LOAD_DIR_NAME}" ]]; then
  echo "missing static load symlink: ${LOAD_PARENT}/${CKPT_LOAD_DIR_NAME}" >&2
  exit 2
fi
zero_files="$(find -L "${SOURCE_CKPT_ITER}" -maxdepth 1 -type f -size 0 | wc -l | tr -d ' ')"
if [[ "${zero_files}" != "0" ]]; then
  echo "source checkpoint has zero-byte files: ${zero_files}" >&2
  exit 2
fi

export CUDA_DEVICE_MAX_CONNECTIONS
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF PYTORCH_CUDA_ALLOC_CONF OMP_NUM_THREADS
export NCCL_DEBUG NCCL_IB_DISABLE NCCL_SOCKET_IFNAME GLOO_SOCKET_IFNAME NCCL_IB_GID_INDEX
export TORCH_NCCL_ENABLE_MONITORING TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC
export TORCHDYNAMO_DISABLE TORCH_COMPILE_DISABLE TORCHDYNAMO_SUPPRESS_ERRORS
export CUDA_VISIBLE_DEVICES
export HF_HOME="${HF_HOME:-${CACHE_DIR}/hf-home}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${CACHE_DIR}/hf-datasets}"
export TMPDIR="${TMPDIR:-${RUN_ROOT}/tmp/node_${NODE_RANK_EARLY}}"
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${TMPDIR}"

NPROC_PER_NODE="${NPROC_PER_NODE:?NPROC_PER_NODE is required}"
NNODES="${NNODES:-${WORLD_SIZE:-1}}"
NODE_RANK="${NODE_RANK:-${RANK:-0}}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:?MASTER_PORT is required}"
PYTHON_BIN="${PYTHON_BIN:?PYTHON_BIN is required}"
export PYTHON_BIN
ATTENTION_MODE="${ATTENTION_MODE:?ATTENTION_MODE is required}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:?ATTENTION_BACKEND is required}"
export ATTENTION_MODE

CONDA_PREFIX_FROM_PY="$("${PYTHON_BIN}" - <<'PY'
import sys
print(sys.prefix)
PY
)"
PY_SITE_PACKAGES="$("${PYTHON_BIN}" - <<'PY'
import site
paths = [p for p in site.getsitepackages() if p.endswith("site-packages")]
print(paths[0])
PY
)"
export PYTHONNOUSERSITE=1
if [[ "${ATTENTION_MODE}" == "fa2" && -n "${FA2_ACTIVATE_SCRIPT:-}" && -f "${FA2_ACTIVATE_SCRIPT}" ]]; then
  # Match the FP8 reference run's Megatron/TE bootstrap before adding our run-specific paths.
  # shellcheck disable=SC1091
  source "${FA2_ACTIVATE_SCRIPT}" >/dev/null
fi
export TRANSFORMER_ENGINE_HOME NVTE_FRAMEWORK NVTE_CUDA_ARCHS FA3_OVERLAY
export NVTE_FLASH_ATTN NVTE_FUSED_ATTN NVTE_UNFUSED_ATTN
if [[ "${ATTENTION_MODE}" == "fa2" ]]; then
  export MEGATRON_FORCE_FLASH_ATTN2=1
  FLASH_ATTN_COMPAT_DIR="${FLASH_ATTN_COMPAT_DIR:?FLASH_ATTN_COMPAT_DIR is required for FA2}"
  test -f "${FLASH_ATTN_COMPAT_DIR}/sitecustomize.py"
  export PYTHONPATH="${FLASH_ATTN_COMPAT_DIR}:${TRANSFORMER_ENGINE_HOME}:${PY_SITE_PACKAGES}:${REPO}:${PYTHONPATH:-}"
elif [[ "${ATTENTION_MODE}" == "fa3" ]]; then
  unset MEGATRON_FORCE_FLASH_ATTN2
  TRANSFORMER_ENGINE_HOME="${FA3_TRANSFORMER_ENGINE_HOME:-${PY_SITE_PACKAGES}}"
  export TRANSFORMER_ENGINE_HOME
  test -f "${FA3_OVERLAY}/flash_attn_interface.py"
  export PYTHONPATH="${FA3_OVERLAY}:${TRANSFORMER_ENGINE_HOME}:${PY_SITE_PACKAGES}:${REPO}:${PYTHONPATH:-}"
else
  echo "unsupported ATTENTION_MODE=${ATTENTION_MODE}" >&2
  exit 2
fi
export PATH="${CONDA_PREFIX_FROM_PY}/bin:${PATH}"
export LD_LIBRARY_PATH="${CONDA_PREFIX_FROM_PY}/lib:${LD_LIBRARY_PATH:-}"
PYTHON_INCLUDE_FROM_ENV="$("${PYTHON_BIN}" - <<'PY'
import sysconfig
print(sysconfig.get_path("include"))
PY
)"
PYTHON_HEADER_OVERLAY="${PYTHON_HEADER_OVERLAY:-${PYTHON_INCLUDE_FROM_ENV}}"
if [[ ! -f "${PYTHON_HEADER_OVERLAY}/Python.h" || ! -f "${PYTHON_HEADER_OVERLAY}/cpython/picklebufobject.h" || ! -f "${PYTHON_HEADER_OVERLAY}/pystats.h" ]]; then
  PYTHON_HEADER_OVERLAY="${PYTHON_INCLUDE_FROM_ENV}"
fi
for py_header_required in "${PYTHON_HEADER_OVERLAY}/Python.h" "${PYTHON_HEADER_OVERLAY}/cpython/picklebufobject.h" "${PYTHON_HEADER_OVERLAY}/pystats.h"; do
  if [[ ! -f "${py_header_required}" ]]; then
    echo "missing required Python header for Triton build: ${py_header_required}" >&2
    exit 2
  fi
done
export CPATH="${PYTHON_HEADER_OVERLAY}${CPATH:+:${CPATH}}"
export C_INCLUDE_PATH="${PYTHON_HEADER_OVERLAY}${C_INCLUDE_PATH:+:${C_INCLUDE_PATH}}"
export CPLUS_INCLUDE_PATH="${PYTHON_HEADER_OVERLAY}${CPLUS_INCLUDE_PATH:+:${CPLUS_INCLUDE_PATH}}"
echo "python_bin=${PYTHON_BIN}"
echo "python_prefix=${CONDA_PREFIX_FROM_PY}"
echo "python_site_packages=${PY_SITE_PACKAGES}"
echo "python_path=${PYTHONPATH}"
echo "torch_compile_env=TORCHDYNAMO_DISABLE:${TORCHDYNAMO_DISABLE},TORCH_COMPILE_DISABLE:${TORCH_COMPILE_DISABLE},TORCHDYNAMO_SUPPRESS_ERRORS:${TORCHDYNAMO_SUPPRESS_ERRORS}"
echo "python_header_overlay=${PYTHON_HEADER_OVERLAY} has_picklebuf=$(test -f "${PYTHON_HEADER_OVERLAY}/cpython/picklebufobject.h" && echo 1 || echo 0) has_pystats=$(test -f "${PYTHON_HEADER_OVERLAY}/pystats.h" && echo 1 || echo 0)"

for required_name in \
  NUM_LAYERS HIDDEN_SIZE FFN_HIDDEN_SIZE NUM_ATTENTION_HEADS NUM_QUERY_GROUPS \
  KV_CHANNELS VOCAB_SIZE MAKE_VOCAB_SIZE_DIVISIBLE_BY MAX_POSITION_EMBEDDINGS \
  ROTARY_BASE ROTARY_PERCENT WINDOW_SIZE WINDOW_ATTN_SKIP_FREQ NORM_EPSILON SEED \
  TP_SIZE PP_SIZE CP_SIZE VIRTUAL_PIPELINE_LAYERS MICRO_BATCH_SIZE GLOBAL_BATCH_SIZE \
  SEQ_LENGTH EPOCH_ITERS TRAIN_ITERS SAVE_INTERVAL EVAL_INTERVAL EVAL_ITERS \
  DATASET_BUILDER_THREADS NUM_WORKERS DATALOADER_TYPE DISTRIBUTED_TIMEOUT_MINUTES \
  RECOMPUTE_GRANULARITY RECOMPUTE_METHOD RECOMPUTE_NUM_LAYERS RECOMPUTE_MODE \
  ENABLE_GRAD_REDUCE_OVERLAP ENABLE_PARAM_GATHER_OVERLAP ENABLE_TP_COMM_OVERLAP \
  ENABLE_P2P_OVERLAP ENABLE_GRAD_ACCUM_FUSION ENABLE_CKPT_FULLY_PARALLEL_LOAD \
  ENABLE_FP8_PARAM_GATHER ENABLE_SAVE; do
  if [[ -z "${!required_name:-}" ]]; then
    echo "missing required training configuration: ${required_name}" >&2
    exit 2
  fi
done

RECOMPUTE_ARGS=()
if [[ "${RECOMPUTE_MODE}" == "full" ]]; then
  RECOMPUTE_ARGS=(
    --recompute-granularity "${RECOMPUTE_GRANULARITY}"
    --recompute-method "${RECOMPUTE_METHOD}"
    --recompute-num-layers "${RECOMPUTE_NUM_LAYERS}"
  )
elif [[ "${RECOMPUTE_MODE}" == "selective" ]]; then
  RECOMPUTE_ARGS=(--recompute-granularity selective)
elif [[ "${RECOMPUTE_MODE}" != "none" ]]; then
  echo "unsupported RECOMPUTE_MODE=${RECOMPUTE_MODE}" >&2
  exit 2
fi

TOTAL_WORLD_SIZE=$((NNODES * NPROC_PER_NODE))
MODEL_PARALLEL_SIZE=$((TP_SIZE * PP_SIZE * CP_SIZE))
if (( TOTAL_WORLD_SIZE % MODEL_PARALLEL_SIZE != 0 )); then
  echo "world size ${TOTAL_WORLD_SIZE} is not divisible by model parallel size ${MODEL_PARALLEL_SIZE}" >&2
  exit 2
fi
DATA_PARALLEL_SIZE=$((TOTAL_WORLD_SIZE / MODEL_PARALLEL_SIZE))
if (( DATA_PARALLEL_SIZE == 1 )); then
  if [[ "${ENABLE_GRAD_REDUCE_OVERLAP}" == "1" || "${ENABLE_PARAM_GATHER_OVERLAP}" == "1" ]]; then
    echo "data_parallel_size=1: disabling DP grad-reduce and param-gather overlap"
  fi
  ENABLE_GRAD_REDUCE_OVERLAP=0
  ENABLE_PARAM_GATHER_OVERLAP=0
fi

OVERLAP_ARGS=()
if [[ "${ENABLE_GRAD_REDUCE_OVERLAP}" == "1" ]]; then
  OVERLAP_ARGS+=(--overlap-grad-reduce)
fi
if [[ "${ENABLE_PARAM_GATHER_OVERLAP}" == "1" ]]; then
  OVERLAP_ARGS+=(--overlap-param-gather)
fi
if [[ "${ENABLE_TP_COMM_OVERLAP}" == "1" ]]; then
  OVERLAP_ARGS+=(--tp-comm-overlap)
  if [[ -n "${TP_COMM_OVERLAP_CFG}" ]]; then
    OVERLAP_ARGS+=(--tp-comm-overlap-cfg "${TP_COMM_OVERLAP_CFG}")
  fi
fi

P2P_ARGS=()
if [[ "${ENABLE_P2P_OVERLAP}" == "0" ]]; then
  P2P_ARGS=(--no-overlap-p2p-communication)
elif [[ "${ENABLE_P2P_OVERLAP}" != "1" ]]; then
  echo "unsupported ENABLE_P2P_OVERLAP=${ENABLE_P2P_OVERLAP}" >&2
  exit 2
fi

VPP_ARGS=()
if [[ "${VIRTUAL_PIPELINE_LAYERS}" -gt 0 ]]; then
  VPP_ARGS=(--num-layers-per-virtual-pipeline-stage "${VIRTUAL_PIPELINE_LAYERS}")
fi

GRAD_ACCUM_ARGS=(--no-gradient-accumulation-fusion)
if [[ "${ENABLE_GRAD_ACCUM_FUSION}" == "1" ]]; then
  GRAD_ACCUM_ARGS=()
fi

CKPT_LOAD_ARGS=()
if [[ "${ENABLE_CKPT_FULLY_PARALLEL_LOAD}" == "1" ]]; then
  CKPT_LOAD_ARGS=(
    --ckpt-fully-parallel-load
    --ckpt-fully-parallel-load-exchange-algo broadcast
  )
fi

FP8_PARAM_GATHER_ARGS=()
if [[ "${ENABLE_FP8_PARAM_GATHER}" == "1" ]]; then
  FP8_PARAM_GATHER_ARGS=(--fp8-param-gather)
fi

SAVE_ARGS=()
if [[ "${ENABLE_SAVE}" == "1" ]]; then
  SAVE_ARGS=(--save "${CKPT_DIR}" --save-interval "${SAVE_INTERVAL}")
elif [[ "${ENABLE_SAVE}" != "0" ]]; then
  echo "unsupported ENABLE_SAVE=${ENABLE_SAVE}" >&2
  exit 2
fi

for required_name in \
  LR MIN_LR LR_DECAY_STYLE LR_WARMUP_ITERS OPTIMIZER WEIGHT_DECAY \
  ADAM_BETA1 ADAM_BETA2 ADAM_EPS CLIP_GRAD MUON_MOMENTUM MUON_SCALE_MODE \
  MUON_FP32_MATMUL_PREC MUON_COEFFICIENT_TYPE MUON_NUM_NS_STEPS MUON_TP_MODE \
  MUON_EXTRA_SCALE_FACTOR MUON_SCALAR_OPTIMIZER; do
  if [[ -z "${!required_name:-}" ]]; then
    echo "missing required optimizer configuration: ${required_name}" >&2
    exit 2
  fi
done

LR_DECAY_ITERS="${LR_DECAY_ITERS:-${TRAIN_ITERS}}"

if [[ "${MEGATRON_GPT_DATASET_SEQUENTIAL:-0}" == "1" ]]; then
  echo "WARNING: MEGATRON_GPT_DATASET_SEQUENTIAL=1 would disable GPT sequential shuffle path; unset for this SFT run"
  unset MEGATRON_GPT_DATASET_SEQUENTIAL
fi

LOCAL_NODE_RANK="${NODE_RANK}"
TELEMETRY_FILE="${LOG_DIR}/${RUN_NAME}_gpu_telemetry_node${LOCAL_NODE_RANK}_$(date +%Y%m%d_%H%M%S).csv"
nvidia-smi --query-gpu=timestamp,index,power.draw,temperature.gpu,utilization.gpu,utilization.memory,memory.used --format=csv -l 10 > "${TELEMETRY_FILE}" 2>&1 &
TELEMETRY_PID="$!"

cd "${REPO}"
LOG_PATH="${LOG_DIR}/${RUN_NAME}_node${LOCAL_NODE_RANK}.log"
echo "run_root=${RUN_ROOT}"
echo "source_ckpt_iter=${SOURCE_CKPT_ITER}"
echo "ckpt_load_iter=${CKPT_LOAD_ITER}"
echo "log_path=${LOG_PATH}"
echo "rank=${NODE_RANK}/${NNODES} master=${MASTER_ADDR}:${MASTER_PORT}"
echo "source_data_args=${SOURCE_DATA_ARGS}"
echo "qcore_root=${QCORE_ROOT}
think_root=${THINK_ROOT}
tool_root=${TOOL_ROOT}
mix_weights=qcore:${MIX_TRAIN_WEIGHT_QCORE},think:${MIX_TRAIN_WEIGHT_THINK},tool:${MIX_TRAIN_WEIGHT_TOOL}"
echo "qcore_train_records=${QCORE_TRAIN_RECORDS} qcore_train_tokens=${QCORE_TRAIN_TOKENS} think_train_records=${THINK_TRAIN_RECORDS} think_train_tokens=${THINK_TRAIN_TOKENS} tool_train_records=${TOOL_TRAIN_RECORDS} tool_raw_records=${TOOL_RAW_RECORDS} tool_train_tokens=${TOOL_TRAIN_TOKENS}"
echo "epoch_iters=${EPOCH_ITERS} train_iters=${TRAIN_ITERS} save_interval=${SAVE_INTERVAL} eval_interval=${EVAL_INTERVAL}"
echo "gbs=${GLOBAL_BATCH_SIZE} seq=${SEQ_LENGTH} tp=${TP_SIZE} pp=${PP_SIZE} cp=${CP_SIZE} vpp_layers=${VIRTUAL_PIPELINE_LAYERS} mbs=${MICRO_BATCH_SIZE}"
echo "ckpt_fully_parallel_load=${ENABLE_CKPT_FULLY_PARALLEL_LOAD}"
echo "fp8_param_gather=${ENABLE_FP8_PARAM_GATHER}"
echo "attention_backend=${ATTENTION_BACKEND} backend_env=flash:${NVTE_FLASH_ATTN},fused:${NVTE_FUSED_ATTN},unfused:${NVTE_UNFUSED_ATTN}"
echo "comm_overlap=grad:${ENABLE_GRAD_REDUCE_OVERLAP},param:${ENABLE_PARAM_GATHER_OVERLAP},tp:${ENABLE_TP_COMM_OVERLAP},tp_cfg:${TP_COMM_OVERLAP_CFG:-none},p2p:${ENABLE_P2P_OVERLAP},gaf:${ENABLE_GRAD_ACCUM_FUSION}"
echo "optimizer=${OPTIMIZER} lr=${LR} min_lr=${MIN_LR} decay=${LR_DECAY_STYLE} warmup=${LR_WARMUP_ITERS} weight_decay=${WEIGHT_DECAY}"
echo "lr_decay_iters=${LR_DECAY_ITERS}"
echo "muon_momentum=${MUON_MOMENTUM} muon_scale_mode=${MUON_SCALE_MODE} muon_fp32_matmul_prec=${MUON_FP32_MATMUL_PREC} muon_coefficient_type=${MUON_COEFFICIENT_TYPE} muon_num_ns_steps=${MUON_NUM_NS_STEPS} muon_tp_mode=${MUON_TP_MODE} muon_extra_scale_factor=${MUON_EXTRA_SCALE_FACTOR} muon_scalar_optimizer=${MUON_SCALAR_OPTIMIZER}"
echo "checkpoint_file_count=$(find -L "${SOURCE_CKPT_ITER}" -maxdepth 1 -type f | wc -l | tr -d ' ') zero_files=${zero_files}"

if [[ "${NODE_RANK}" == "0" && "${ENABLE_SWANLAB:-0}" == "1" ]]; then
  (
    : "${SWANLAB_TAIL_SCRIPT:?SWANLAB_TAIL_SCRIPT is required when ENABLE_SWANLAB=1}"
    if [[ -f "${PROJECT_ROOT}/.secrets/swanlab.env" ]]; then
      set -a
      # shellcheck disable=SC1091
      source "${PROJECT_ROOT}/.secrets/swanlab.env"
      set +a
    fi
    if [[ -n "${HTTP_PROXY:-}" ]]; then
      export HTTP_PROXY
      export http_proxy="${http_proxy:-${HTTP_PROXY}}"
    fi
    if [[ -n "${HTTPS_PROXY:-}" ]]; then
      export HTTPS_PROXY
      export https_proxy="${https_proxy:-${HTTPS_PROXY}}"
    fi
    SWANLAB_PYTHON="${SWANLAB_PYTHON:-${PROJECT_ROOT}/swanlab_venv/bin/python}"
    if [[ ! -x "${SWANLAB_PYTHON}" ]]; then
      SWANLAB_PYTHON="${PYTHON_BIN}"
    fi
    "${SWANLAB_PYTHON}" "${SWANLAB_TAIL_SCRIPT}" \
      --log-path "${LOG_PATH}" \
      --project "${SWANLAB_PROJECT:-zgcm-s4-sft}" \
      --run-name "${SWANLAB_RUN_NAME:-${RUN_NAME}}" \
      --config-json "${CONFIG_DIR}/run_manifest.json"
  ) > "${LOG_DIR}/${RUN_NAME}_swanlab_tail.log" 2>&1 &
  SWANLAB_PID="$!"
fi


"${PYTHON_BIN}" - <<'PY'
import importlib
import importlib.metadata as md
import os
import sys

mods = ["torch", "transformer_engine", "transformer_engine.pytorch", "emerging_optimizers"]
for mod in mods:
    module = importlib.import_module(mod)
    print(f"import_ok {mod} {getattr(module, '__file__', '<no-file>')}")
for pkg in ("transformer-engine", "transformer-engine-cu12", "transformer-engine-torch", "emerging-optimizers", "flash-attn"):
    try:
        print(f"pkg_version {pkg} {md.version(pkg)}")
    except md.PackageNotFoundError:
        print(f"pkg_version {pkg} <missing>")
attention_mode = os.environ["ATTENTION_MODE"]
if attention_mode == "fa2":
    try:
        md.version("flash-attn-3")
    except md.PackageNotFoundError:
        print("flash_attn3_hidden=ok")
    else:
        raise RuntimeError("flash-attn-3 must be hidden in FA2 mode")
    from flash_attn import flash_attn_varlen_func
    assert callable(flash_attn_varlen_func)
    print("flash_attn2_varlen_callable=ok")
elif attention_mode == "fa3":
    print("flash_attn3_version", md.version("flash-attn-3"))
    from flash_attn_interface import flash_attn_varlen_func
    assert callable(flash_attn_varlen_func)
    print("flash_attn3_varlen_callable=ok")
else:
    raise RuntimeError(f"unexpected attention mode: {attention_mode}")
print("python_executable", sys.executable)
print("python_sys_path_head", sys.path[:6])
from megatron.core import jit as megatron_jit
from megatron.core.dist_checkpointing.strategies import torch as dcp_torch
import torch

@megatron_jit.jit_fuser
def _jit_probe(x):
    return x + 1

print("megatron_jit_probe", _jit_probe(torch.ones(2)).tolist())
print("dcp_global_gloo_patch", "_get_dcp_global_gloo_group" in open(dcp_torch.__file__, "r", encoding="utf-8").read())
print("dcp_have_nvrx", getattr(dcp_torch, "HAVE_NVRX", None))
dcp_torch.get_async_strategy("mcore")
print("dcp_mcore_strategy=ok")
print("python_import_gate=ok")
PY

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1: generated load symlink, data args, manifest, and environment checks; not launching torchrun."
  exit 0
fi

export MEGATRON_DCP_FORCE_MCORE_SAVE="${MEGATRON_DCP_FORCE_MCORE_SAVE:-1}"
export MEGATRON_DCP_SAVE_ASYNC_STRATEGY="${MEGATRON_DCP_SAVE_ASYNC_STRATEGY:-mcore}"
export MEGATRON_ENABLE_GLM51_SFT_PATCH="${MEGATRON_ENABLE_GLM51_SFT_PATCH:-1}"
export MEGATRON_GLM51_SFT_TOKENIZER_FILE="${MEGATRON_GLM51_SFT_TOKENIZER_FILE:-${REPO}/megatron/core/tokenizers/text/libraries/sft_tokenizer.py}"
export MEGATRON_ENABLE_INDEXED_SFT_DATASET_PATCH="${MEGATRON_ENABLE_INDEXED_SFT_DATASET_PATCH:-1}"
export MEGATRON_GLM51_SFT_DATASET_FILE="${MEGATRON_GLM51_SFT_DATASET_FILE:-${REPO}/megatron/training/datasets/sft_dataset.py}"

"${PYTHON_BIN}" -m torch.distributed.run \
  --nproc_per_node "${NPROC_PER_NODE}" \
  --nnodes "${NNODES}" \
  --node_rank "${NODE_RANK}" \
  --master_addr "${MASTER_ADDR}" \
  --master_port "${MASTER_PORT}" \
  --no-python \
  "${SCRIPT_DIR}/pretrain_gpt_rank_cache_mcore_wrapper.sh" \
  --use-mcore-models \
  --num-layers "${NUM_LAYERS}" \
  --hidden-size "${HIDDEN_SIZE}" \
  --ffn-hidden-size "${FFN_HIDDEN_SIZE}" \
  --num-attention-heads "${NUM_ATTENTION_HEADS}" \
  --kv-channels "${KV_CHANNELS}" \
  --seq-length "${SEQ_LENGTH}" \
  --max-position-embeddings "${MAX_POSITION_EMBEDDINGS}" \
  --position-embedding-type rope \
  --rotary-base "${ROTARY_BASE}" \
  --rotary-percent "${ROTARY_PERCENT}" \
  --attention-dropout 0.0 \
  --hidden-dropout 0.0 \
  --swiglu \
  --normalization RMSNorm \
  --disable-bias-linear \
  --untie-embeddings-and-output-weights \
  --attention-backend "${ATTENTION_BACKEND}" \
  --make-vocab-size-divisible-by "${MAKE_VOCAB_SIZE_DIVISIBLE_BY}" \
  --group-query-attention \
  --num-query-groups "${NUM_QUERY_GROUPS}" \
  --window-size "${WINDOW_SIZE}" \
  --window-attn-skip-freq "${WINDOW_ATTN_SKIP_FREQ}" \
  --norm-epsilon "${NORM_EPSILON}" \
  --qk-layernorm \
  --attention-output-gate \
  --attention-output-gate-only-swa \
  --transformer-impl transformer_engine \
  --fp8-format hybrid \
  --fp8-recipe delayed \
  --fp8-amax-compute-algo max \
  --fp8-amax-history-len 1024 \
  "${FP8_PARAM_GATHER_ARGS[@]}" \
  --no-rope-fusion \
  --micro-batch-size "${MICRO_BATCH_SIZE}" \
  --global-batch-size "${GLOBAL_BATCH_SIZE}" \
  --optimizer "${OPTIMIZER}" \
  --lr "${LR}" \
  --min-lr "${MIN_LR}" \
  --lr-decay-style "${LR_DECAY_STYLE}" \
  --weight-decay "${WEIGHT_DECAY}" \
  --adam-beta1 "${ADAM_BETA1}" \
  --adam-beta2 "${ADAM_BETA2}" \
  --adam-eps "${ADAM_EPS}" \
  --clip-grad "${CLIP_GRAD}" \
  --muon-momentum "${MUON_MOMENTUM}" \
  --muon-scale-mode "${MUON_SCALE_MODE}" \
  --muon-fp32-matmul-prec "${MUON_FP32_MATMUL_PREC}" \
  --muon-coefficient-type "${MUON_COEFFICIENT_TYPE}" \
  --muon-num-ns-steps "${MUON_NUM_NS_STEPS}" \
  --muon-tp-mode "${MUON_TP_MODE}" \
  --muon-extra-scale-factor "${MUON_EXTRA_SCALE_FACTOR}" \
  --muon-scalar-optimizer "${MUON_SCALAR_OPTIMIZER}" \
  --bf16 \
  --calculate-per-token-loss \
  --seed "${SEED}" \
  --train-iters "${TRAIN_ITERS}" \
  --lr-decay-iters "${LR_DECAY_ITERS}" \
  --lr-warmup-iters "${LR_WARMUP_ITERS}" \
  "${RECOMPUTE_ARGS[@]}" \
  --finetune \
  --no-load-optim \
  --no-load-rng \
  --use-distributed-optimizer \
  --tensor-model-parallel-size "${TP_SIZE}" \
  --pipeline-model-parallel-size "${PP_SIZE}" \
  --context-parallel-size "${CP_SIZE}" \
  "${VPP_ARGS[@]}" \
  --sequence-parallel \
  "${OVERLAP_ARGS[@]}" \
  "${P2P_ARGS[@]}" \
  "${GRAD_ACCUM_ARGS[@]}" \
  --vocab-size "${VOCAB_SIZE}" \
  --data-cache-path "${CACHE_DIR}" \
  --no-create-attention-mask-in-dataloader \
  --num-workers "${NUM_WORKERS}" \
  --dataloader-type "${DATALOADER_TYPE}" \
  --num-dataset-builder-threads "${DATASET_BUILDER_THREADS}" \
  --mid-level-dataset-surplus 1.0 \
  --sft \
  --tokenizer-type SFTTokenizer \
  --tokenizer-model "${TOKENIZER}" \
  --sft-tokenizer-prompt-format glm51 \
  --per-split-data-args-path "${DATA_DIR}/per_split_data_args.json" \
  --log-interval 1 \
  --eval-interval "${EVAL_INTERVAL}" \
  --eval-iters "${EVAL_ITERS}" \
  --log-throughput \
  --ckpt-format torch_dist \
  --async-strategy mcore \
  "${CKPT_LOAD_ARGS[@]}" \
  --distributed-timeout-minutes "${DISTRIBUTED_TIMEOUT_MINUTES}" \
  --tensorboard-dir "${TB_DIR}" \
  "${SAVE_ARGS[@]}" \
  --load "${LOAD_PARENT}" \
  2>&1 | tee "${LOG_PATH}"

if [[ "${NODE_RANK}" == "0" ]]; then
  touch "${STATUS_DIR}/train.done"
  touch "${PROJECT_ROOT}/status/train.done"
fi
