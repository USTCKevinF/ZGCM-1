"""Disjoint prefix-sampled Midtrain mix construction.

Shorter base buckets are eligible for longer final mixes, but an ID selected
for an earlier final mix is removed from all later candidate pools. This
keeps the final mixes disjoint while preserving prefix eligibility.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence


DEFAULT_ELIGIBLE = {
    "mix16": ("B16",),
    "mix64": ("B16", "B64"),
    "mix256": ("B16", "B64", "B256"),
}


def _rank(value: str, seed: str) -> bytes:
    return hashlib.blake2b(f"{seed}\0{value}".encode("utf-8"), digest_size=16).digest()


def prefix_sample_mix(
    base_buckets: Mapping[str, Sequence[str]],
    final_targets: Mapping[str, int],
    *,
    seed: str = "zgcm-midtrain",
    eligible_by_final: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, list[str]]:
    """Select disjoint final mixes from nested prefix-eligible base buckets.

    ``final_targets`` is processed in the canonical order ``mix16``,
    ``mix64``, ``mix256``. A custom ``eligible_by_final`` can be supplied for
    another stage configuration, but each final mix still excludes all IDs
    selected by earlier mixes.
    """
    eligibility = eligible_by_final or DEFAULT_ELIGIBLE
    selected: set[str] = set()
    result: dict[str, list[str]] = {}
    order = [name for name in ("mix16", "mix64", "mix256") if name in final_targets]
    order.extend(name for name in final_targets if name not in order)
    for final_name in order:
        target = int(final_targets[final_name])
        if target < 0:
            raise ValueError("final mix targets must be non-negative")
        candidates = {
            str(record_id)
            for bucket in eligibility.get(final_name, ())
            for record_id in base_buckets.get(bucket, ())
            if str(record_id) not in selected
        }
        if len(candidates) < target:
            raise ValueError(f"{final_name} needs {target} samples but only {len(candidates)} are available")
        chosen = sorted(candidates, key=lambda record_id: (_rank(record_id, f"{seed}:{final_name}"), record_id))[:target]
        result[final_name] = chosen
        selected.update(chosen)
    return result


__all__ = ["DEFAULT_ELIGIBLE", "prefix_sample_mix"]
