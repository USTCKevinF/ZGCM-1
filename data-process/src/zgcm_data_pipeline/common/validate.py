"""Stage-neutral validation of released records."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any


REQUIRED_RECORD_KEYS = {"dataset", "source_id", "stage", "category", "decision", "data"}


def validate_release_record(record: Mapping[str, Any]) -> None:
    """Raise ``ValueError`` when a serialized record violates the release contract."""
    missing = REQUIRED_RECORD_KEYS.difference(record)
    if missing:
        raise ValueError(f"record is missing required keys: {sorted(missing)}")
    if record["decision"] not in {"keep", "review", "drop"}:
        raise ValueError(f"unsupported decision: {record['decision']!r}")
    if not isinstance(record["data"], Mapping):
        raise ValueError("record.data must be an object")


def summarize_decisions(records: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    """Count release decisions without retaining sample content."""
    counts = Counter(str(record.get("decision", "unknown")) for record in records)
    return dict(sorted(counts.items()))


__all__ = ["summarize_decisions", "validate_release_record"]
