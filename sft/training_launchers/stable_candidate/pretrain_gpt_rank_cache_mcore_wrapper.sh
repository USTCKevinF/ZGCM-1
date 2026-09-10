#!/usr/bin/env bash
set -euo pipefail

rank="${RANK:-0}"
local_rank="${LOCAL_RANK:-0}"
cache_key="r${rank}l${local_rank}"
cache_root="${SFT_CACHE_ROOT:-${TMPDIR:?TMPDIR is required}}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${cache_root}/ti_${cache_key}}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${cache_root}/tr_${cache_key}}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${cache_root}/xd_${cache_key}}"
export TMPDIR="${SFT_RANK_TMPDIR:-${cache_root}/t_${cache_key}}"
export TMP="${TMP:-${TMPDIR}}"
export TEMP="${TEMP:-${TMPDIR}}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-${cache_root}/cu_${cache_key}}"
mkdir -p "${TORCHINDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}" "${XDG_CACHE_HOME}" "${TMPDIR}" "${CUDA_CACHE_PATH}"

PYTHON_BIN="${PYTHON_BIN:?PYTHON_BIN is required}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${PYTHON_BIN}" -u "${SCRIPT_DIR}/pretrain_gpt_force_mcore_save.py" "$@"
