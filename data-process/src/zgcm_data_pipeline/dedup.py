from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from dataclasses import dataclass, field


_TOKEN = re.compile(r"\w+", re.UNICODE)


def simhash64(text: str, *, ngram: int = 5) -> int:
    """Return a deterministic 64-bit SimHash over normalized word n-grams."""
    tokens = _TOKEN.findall(text.casefold())
    if not tokens:
        return 0
    width = max(1, int(ngram))
    features = (" ".join(tokens[index : index + width]) for index in range(max(1, len(tokens) - width + 1)))
    weights = [0] * 64
    for feature in features:
        digest = int.from_bytes(hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest(), "big")
        for bit in range(64):
            weights[bit] += 1 if digest & (1 << bit) else -1
    result = 0
    for bit, weight in enumerate(weights):
        if weight >= 0:
            result |= 1 << bit
    return result


def hamming_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


@dataclass(slots=True)
class NearDeduper:
    """Streaming SimHash deduper using band indexes to avoid all-pairs scans.

    With four bands and a distance threshold of at most three, any matching
    pair must share at least one band, so candidate lookup has no false
    negatives relative to the configured SimHash distance.
    """

    threshold: int = 3
    bands: int = 4
    fingerprints: list[int] = field(default_factory=list)
    indexes: list[dict[int, set[int]]] = field(init=False)

    def __post_init__(self) -> None:
        if 64 % self.bands != 0 or self.threshold >= self.bands:
            raise ValueError("bands must divide 64 and threshold must be smaller than bands")
        self.indexes = [defaultdict(set) for _ in range(self.bands)]

    def _band_values(self, fingerprint: int) -> list[int]:
        width = 64 // self.bands
        mask = (1 << width) - 1
        return [(fingerprint >> (band * width)) & mask for band in range(self.bands)]

    def check_and_add(self, text: str, *, ngram: int = 5) -> bool:
        fingerprint = simhash64(text, ngram=ngram)
        values = self._band_values(fingerprint)
        candidates: set[int] = set()
        for band, value in enumerate(values):
            candidates.update(self.indexes[band].get(value, ()))
        duplicate = any(
            hamming_distance(fingerprint, self.fingerprints[index]) <= self.threshold
            for index in candidates
        )
        index = len(self.fingerprints)
        self.fingerprints.append(fingerprint)
        for band, value in enumerate(values):
            self.indexes[band][value].add(index)
        return duplicate
