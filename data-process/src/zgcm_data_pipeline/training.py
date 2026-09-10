from __future__ import annotations

import bisect
import hashlib
import math
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence


IGNORE_INDEX = -100


@dataclass(frozen=True, slots=True)
class WorkUnit:
    key: str
    weight: int


def balance_work_units(units: Sequence[WorkUnit], workers: int) -> list[list[WorkUnit]]:
    """Assign heavy units first to the currently lightest worker (LPT)."""
    if workers <= 0:
        raise ValueError("workers must be positive")
    groups: list[list[WorkUnit]] = [[] for _ in range(workers)]
    loads = [(0, worker) for worker in range(workers)]
    for unit in sorted(units, key=lambda item: (-item.weight, item.key)):
        load, worker = loads.pop(0)
        groups[worker].append(unit)
        bisect.insort(loads, (load + max(0, int(unit.weight)), worker))
    return groups


def stable_partition(record_key: str, partitions: int, *, seed: str = "zgcm") -> int:
    """Place one sample in a deterministic shuffle partition."""
    if partitions <= 0:
        raise ValueError("partitions must be positive")
    digest = hashlib.blake2b(f"{seed}\0{record_key}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % partitions


def deterministic_shuffle_key(record_key: str, *, seed: str = "zgcm") -> bytes:
    """Return a stable sort key for reproducible sample-level shuffling."""
    return hashlib.blake2b(f"{seed}\0{record_key}".encode("utf-8"), digest_size=16).digest()


def allocate_mix_quotas(weights: Mapping[str, float], total_samples: int) -> dict[str, int]:
    """Allocate an exact sample budget with normalized weights and largest remainders."""
    if total_samples < 0:
        raise ValueError("total_samples must be non-negative")
    clean = {str(name): float(weight) for name, weight in weights.items() if float(weight) > 0}
    if not clean:
        raise ValueError("at least one positive mix weight is required")
    total_weight = sum(clean.values())
    exact = {name: total_samples * weight / total_weight for name, weight in clean.items()}
    quotas = {name: math.floor(value) for name, value in exact.items()}
    remaining = total_samples - sum(quotas.values())
    order = sorted(clean, key=lambda name: (-(exact[name] - quotas[name]), name))
    for name in order[:remaining]:
        quotas[name] += 1
    return quotas


def select_mixed_sample_ids(
    source_to_ids: Mapping[str, Iterable[str]],
    quotas: Mapping[str, int],
    *,
    seed: str = "zgcm",
) -> list[str]:
    """Select each source quota and deterministically interleave the resulting mix."""
    selected: list[tuple[bytes, str]] = []
    for source, quota in sorted(quotas.items()):
        if quota < 0:
            raise ValueError("mix quotas must be non-negative")
        ranked = sorted(
            (str(record_id) for record_id in source_to_ids.get(source, ())),
            key=lambda record_id: deterministic_shuffle_key(record_id, seed=f"{seed}:{source}"),
        )
        if len(ranked) < quota:
            raise ValueError(f"source {source!r} has {len(ranked)} samples but quota is {quota}")
        for record_id in ranked[:quota]:
            selected.append((deterministic_shuffle_key(record_id, seed=f"{seed}:mix"), record_id))
    selected.sort(key=lambda item: (item[0], item[1]))
    return [record_id for _, record_id in selected]


@dataclass(frozen=True, slots=True)
class RowMicroshard:
    source: str
    row_start: int
    row_end: int
    tokens: int


def plan_row_microshards(
    source: str,
    row_token_counts: Iterable[int],
    *,
    target_tokens: int,
) -> list[RowMicroshard]:
    """Build contiguous row spans close to a target token budget."""
    if target_tokens <= 0:
        raise ValueError("target_tokens must be positive")
    plans: list[RowMicroshard] = []
    start = 0
    total = 0
    end = 0
    for end, raw_tokens in enumerate(row_token_counts, 1):
        tokens = max(0, int(raw_tokens))
        if total and total + tokens > target_tokens:
            plans.append(RowMicroshard(source, start, end - 1, total))
            start = end - 1
            total = 0
        total += tokens
    if end > start:
        plans.append(RowMicroshard(source, start, end, total))
    return plans


@dataclass(frozen=True, slots=True)
class TokenizedSample:
    tokens: tuple[int, ...]
    source_id: str
    targets: tuple[int, ...] | None = None

    @property
    def length(self) -> int:
        return len(self.tokens)


@dataclass(slots=True)
class PackedSequence:
    tokens: list[int] = field(default_factory=list)
    targets: list[int] | None = None
    cu_seqlens: list[int] = field(default_factory=lambda: [0])
    source_ids: list[str] = field(default_factory=list)


def validate_tokenized_sample(sample: TokenizedSample, *, ignore_index: int = IGNORE_INDEX) -> None:
    if not sample.tokens:
        raise ValueError("tokenized sample is empty")
    if sample.targets is None:
        return
    if len(sample.tokens) != len(sample.targets):
        raise ValueError("tokens and targets must have the same length")
    if not any(target != ignore_index for target in sample.targets):
        raise ValueError("SFT sample has no trainable target token")


def build_loss_mask(targets: Sequence[int], *, ignore_index: int = IGNORE_INDEX) -> tuple[int, ...]:
    return tuple(0 if target == ignore_index else 1 for target in targets)


def pack_tokenized_samples(
    samples: Iterable[TokenizedSample],
    *,
    sequence_length: int,
    allow_overlength: bool = False,
) -> tuple[list[PackedSequence], list[TokenizedSample]]:
    """First-fit-decreasing packing while preserving SFT target boundaries."""
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    accepted: list[TokenizedSample] = []
    overlength: list[TokenizedSample] = []
    for sample in samples:
        validate_tokenized_sample(sample)
        if sample.length > sequence_length:
            overlength.append(sample)
            if not allow_overlength:
                continue
        accepted.append(sample)

    accepted.sort(key=lambda sample: (-sample.length, sample.source_id))
    bins: list[PackedSequence] = []
    free: list[tuple[int, int]] = []
    for sample in accepted:
        position = bisect.bisect_left(free, (sample.length, -1))
        if position == len(free):
            bin_id = len(bins)
            bins.append(PackedSequence(targets=[] if sample.targets is not None else None))
        else:
            _, bin_id = free.pop(position)
        packed = bins[bin_id]
        if (packed.targets is None) != (sample.targets is None):
            raise ValueError("cannot mix pretraining and SFT samples in one packing call")
        packed.tokens.extend(sample.tokens)
        if packed.targets is not None and sample.targets is not None:
            packed.targets.extend(sample.targets)
        packed.source_ids.append(sample.source_id)
        packed.cu_seqlens.append(len(packed.tokens))
        remaining = sequence_length - len(packed.tokens)
        if remaining > 0:
            bisect.insort(free, (remaining, bin_id))
    return bins, overlength
