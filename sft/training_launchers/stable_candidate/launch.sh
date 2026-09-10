#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
env_file="${SFT_ENV_FILE:-${script_dir}/runtime.env}"
config_file="${SFT_CONFIG_FILE:-${script_dir}/zgcm.conf}"

for required in "${env_file}" "${config_file}"; do
  if [[ ! -f "${required}" ]]; then
    echo "missing SFT configuration file: ${required}" >&2
    exit 2
  fi
done

set -a
source "${env_file}"
source "${config_file}"
set +a

export REPO="${REPO:-${repo_root}}"
export VARIANT="${VARIANT:-${RUN_NAME}}"
export RUN_NAME="${SFT_RUN_NAME_OVERRIDE:-${RUN_NAME}}"
export VARIANT="${SFT_RUN_NAME_OVERRIDE:-${VARIANT}}"
export PYTHON_BIN="${PYTHON_BIN:-${ENV_ROOT}/bin/python}"

if [[ -n "${SFT_GLOBAL_BATCH_SIZE_OVERRIDE:-}" ]]; then
  export GLOBAL_BATCH_SIZE="${SFT_GLOBAL_BATCH_SIZE_OVERRIDE}"
fi
if [[ -n "${SFT_WORKER_NUM_OVERRIDE:-}" ]]; then
  export WORKER_NUM="${SFT_WORKER_NUM_OVERRIDE}"
fi

for name in DATASET_METADATA EPOCHS SEQ_LENGTH GLOBAL_BATCH_SIZE SAVE_EVERY_EPOCHS; do
  if [[ -z "${!name:-}" ]]; then
    echo "missing required training setting: ${name}" >&2
    exit 2
  fi
done

if [[ ! -f "${DATASET_METADATA}" ]]; then
  echo "missing dataset metadata: ${DATASET_METADATA}" >&2
  exit 2
fi

read -r TRAIN_TOKENS TRAIN_RECORDS < <("${PYTHON_BIN}" - "${DATASET_METADATA}" <<'PY'
import json
import sys

metadata_path = sys.argv[1]
with open(metadata_path, "r", encoding="utf-8") as handle:
    data = json.load(handle)


def first_value(paths, *, required):
    for path in paths:
        value = data
        for key in path:
            if not isinstance(value, dict) or key not in value:
                break
            value = value[key]
        else:
            value = int(value)
            if value > 0:
                return value
    if required:
        rendered = ", ".join(".".join(path) for path in paths)
        raise SystemExit(f"no positive dataset count found in {metadata_path}; tried: {rendered}")
    return 0


tokens = first_value(
    [
        ("train_tokens",),
        ("combined", "train_tokens"),
        ("train", "tokens"),
        ("splits", "train", "tokens"),
        ("split_stats", "train", "tokens"),
    ],
    required=True,
)
records = first_value(
    [
        ("train_records",),
        ("combined", "train_records"),
        ("train", "records"),
        ("splits", "train", "records"),
        ("split_stats", "train", "records"),
    ],
    required=False,
)
print(tokens, records)
PY
)
export TRAIN_TOKENS TRAIN_RECORDS

if [[ -n "${EXPECTED_TRAIN_TOKENS:-}" && "${TRAIN_TOKENS}" != "${EXPECTED_TRAIN_TOKENS}" ]]; then
  echo "dataset train token mismatch: metadata=${TRAIN_TOKENS} expected=${EXPECTED_TRAIN_TOKENS}" >&2
  exit 2
fi
if [[ -n "${EXPECTED_TRAIN_RECORDS:-}" && "${TRAIN_RECORDS}" != "${EXPECTED_TRAIN_RECORDS}" ]]; then
  echo "dataset train record mismatch: metadata=${TRAIN_RECORDS} expected=${EXPECTED_TRAIN_RECORDS}" >&2
  exit 2
fi
export MIX_TOTAL_TRAIN_TOKENS="${TRAIN_TOKENS}"
export MIX_TOTAL_TRAIN_RECORDS="${TRAIN_RECORDS}"

export EPOCH_ITERS="$(${PYTHON_BIN} -c \
  'import sys; t,s,g=map(int,sys.argv[1:]); print(t//(s*g))' \
  "${TRAIN_TOKENS}" "${SEQ_LENGTH}" "${GLOBAL_BATCH_SIZE}")"
export TRAIN_ITERS="$(${PYTHON_BIN} -c \
  'import sys; t,e,s,g=map(int,sys.argv[1:]); print(t*e//(s*g))' \
  "${TRAIN_TOKENS}" "${EPOCHS}" "${SEQ_LENGTH}" "${GLOBAL_BATCH_SIZE}")"
export SAVE_INTERVAL="$((EPOCH_ITERS * SAVE_EVERY_EPOCHS))"
export EVAL_INTERVAL="${EVAL_INTERVAL:-${SAVE_INTERVAL}}"

effective_epochs="$(${PYTHON_BIN} -c \
  'import sys; t,i,s,g=map(int,sys.argv[1:]); print(i*s*g/t)' \
  "${TRAIN_TOKENS}" "${TRAIN_ITERS}" "${SEQ_LENGTH}" "${GLOBAL_BATCH_SIZE}")"
${PYTHON_BIN} -c \
  'import sys; actual,expected=map(float,sys.argv[1:]); assert abs(actual-expected)<0.01,(actual,expected)' \
  "${effective_epochs}" "${EPOCHS}"

planned_train_iters="${TRAIN_ITERS}"
planned_effective_epochs="${effective_epochs}"
export LR_DECAY_ITERS="${LR_DECAY_ITERS:-${planned_train_iters}}"
if [[ -n "${SMOKE_TRAIN_ITERS:-}" ]]; then
  if [[ ! "${SMOKE_TRAIN_ITERS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "SMOKE_TRAIN_ITERS must be a positive integer" >&2
    exit 2
  fi
  if (( SMOKE_TRAIN_ITERS > planned_train_iters )); then
    echo "SMOKE_TRAIN_ITERS exceeds planned train iterations" >&2
    exit 2
  fi
  export TRAIN_ITERS="${SMOKE_TRAIN_ITERS}"
  export ENABLE_SAVE=0
  effective_epochs="$(${PYTHON_BIN} -c \
    'import sys; t,i,s,g=map(int,sys.argv[1:]); print(i*s*g/t)' \
    "${TRAIN_TOKENS}" "${TRAIN_ITERS}" "${SEQ_LENGTH}" "${GLOBAL_BATCH_SIZE}")"
fi

echo "BOOT ${RUN_NAME} $(date -Is) host=$(hostname)"
echo "dataset_metadata=${DATASET_METADATA} train_records=${TRAIN_RECORDS} train_tokens=${TRAIN_TOKENS}"
echo "epochs=${EPOCHS} epoch_iters=${EPOCH_ITERS} planned_train_iters=${planned_train_iters} planned_effective_epochs=${planned_effective_epochs}"
echo "active_train_iters=${TRAIN_ITERS} lr_decay_iters=${LR_DECAY_ITERS} active_effective_epochs=${effective_epochs} save_enabled=${ENABLE_SAVE}"
if [[ "${VALIDATE_CONFIG_ONLY:-0}" == "1" ]]; then
  echo "VALIDATE_CONFIG_ONLY=1: dataset identity and derived training configuration are valid."
  exit 0
fi

runner="${script_dir}/run_sft_thinkmix64k_fa3_tpcomm_formal_20260715.sh"
for required in \
  "${runner}" \
  "${script_dir}/pretrain_gpt_rank_cache_mcore_wrapper.sh" \
  "${script_dir}/pretrain_gpt_force_mcore_save.py"; do
  if [[ ! -f "${required}" ]]; then
    echo "missing frozen training script: ${required}" >&2
    exit 2
  fi
done

exec bash "${runner}"
