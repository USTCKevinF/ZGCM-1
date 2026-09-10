"""Nemotron Math strict hard-error filtering."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import PurePath
from typing import Any

from ....core import PipelineContext, Step
from ....models import Decision, Sample


def record_drop_key(data: Mapping[str, Any]) -> tuple[str, int] | None:
    metadata = data.get("meta") if isinstance(data.get("meta"), Mapping) else data.get("metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    source = metadata.get("source_rel") or metadata.get("source_file")
    if not source and metadata.get("source_path"):
        source = PurePath(str(metadata["source_path"])).name
    row = metadata.get("row_index")
    if row is None:
        row = metadata.get("file_row_index")
    if source is None or row is None:
        return None
    try:
        return str(source), int(row)
    except (TypeError, ValueError):
        return None


def normalize_drop_keys(values: Any) -> set[tuple[str, int]]:
    """Normalize JSON-configurable drop keys into ``(source, row)`` tuples."""
    if not isinstance(values, Iterable) or isinstance(values, (str, bytes, Mapping)):
        return set()
    result: set[tuple[str, int]] = set()
    for value in values:
        if isinstance(value, str):
            parts = value.split("\t") if "\t" in value else value.rsplit(":", 1)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            parts = list(value)
        elif isinstance(value, Mapping):
            row = value.get("file_row_index")
            if row is None:
                row = value.get("row_index")
            parts = [value.get("source_file") or value.get("source_rel"), row]
        else:
            continue
        if len(parts) != 2 or parts[0] is None or parts[1] is None:
            continue
        try:
            result.add((str(parts[0]), int(parts[1])))
        except (TypeError, ValueError):
            continue
    return result


def _hard_error_score(data: Mapping[str, Any]) -> float | None:
    metadata = data.get("meta") if isinstance(data.get("meta"), Mapping) else data.get("metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    for name in ("hard_error_score", "math_hard_error_score", "error_risk_score"):
        value = data.get(name, metadata.get(name))
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def clean_nemotron_math_record(sample: Sample, context: PipelineContext) -> Sample:
    """Drop records listed by lineage key or carrying a strict high-risk score."""
    key = record_drop_key(sample.data)
    configured_keys = normalize_drop_keys(context.config.get("nemotron_math_drop_keys", ()))
    score = _hard_error_score(sample.data)
    threshold = float(context.config.get("nemotron_math_hard_error_threshold", 4.0))
    if key is not None:
        sample.data["nemotron_math_drop_key"] = [key[0], key[1]]
    else:
        sample.flags.add("nemotron_math_missing_lineage_key")
    if score is not None:
        sample.metrics["nemotron_math_hard_error_score"] = score
        sample.buckets["hard_error_risk"] = "high" if score >= threshold else "accepted"
    if (key is not None and key in configured_keys) or (score is not None and score >= threshold):
        sample.mark(Decision.DROP, "nemotron_math_strict_hard_error")
    elif key is None and score is None:
        sample.mark(Decision.REVIEW, "nemotron_math_missing_hard_error_signal")
    return sample


NEMOTRON_MATH_STEPS = (
    Step("clean_nemotron_math_record", clean_nemotron_math_record, "Remove Nemotron Math strict hard errors"),
)


__all__ = [
    "NEMOTRON_MATH_STEPS",
    "clean_nemotron_math_record",
    "normalize_drop_keys",
    "record_drop_key",
]
