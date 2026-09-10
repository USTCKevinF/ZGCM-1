#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 OUTPUT LONG64K_INDEXED_ROOT SHORT16K_INDEXED_ROOT" >&2
  exit 2
fi

output="$1"
long_root="$2"
short_root="$3"
mkdir -p "$(dirname "${output}")"
tmp="${output}.tmp.$$"
trap 'rm -f "${tmp}"' EXIT

: > "${tmp}"
for shard in $(seq 0 511); do
  printf -v id '%05d' "${shard}"
  prefix="${long_root}/shard_${id}/mix64_v3_glm51_shuffle512_shard_${id}_text_document"
  [[ -f "${prefix}.idx" && -f "${prefix}.bin" ]] || { echo "missing ${prefix}.{idx,bin}" >&2; exit 3; }
  printf '2 %s\n' "${prefix}" >> "${tmp}"
done
for shard in $(seq 0 255); do
  printf -v id '%05d' "${shard}"
  prefix="${short_root}/shard_${id}/mix16_v4_glm51_shuffle256_shard_${id}_text_document"
  [[ -f "${prefix}.idx" && -f "${prefix}.bin" ]] || { echo "missing ${prefix}.{idx,bin}" >&2; exit 3; }
  printf '1 %s\n' "${prefix}" >> "${tmp}"
done
mv "${tmp}" "${output}"
trap - EXIT
echo "wrote 768 weighted prefixes to ${output}"
