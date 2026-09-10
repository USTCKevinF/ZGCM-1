"""FinePDFs zh/en quality rules extracted from the collected processing job."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from typing import Any

from ....core import PipelineContext, Step
from ....models import Decision, Sample


def _number(data: dict[str, Any], names: Iterable[str]) -> float | None:
    metadata = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
    for name in names:
        value = data.get(name, metadata.get(name))
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _dedup_key(data: dict[str, Any], text: str) -> str:
    if data.get("id"):
        return f"id:{data['id']}"
    provenance = tuple(str(data.get(name) or "") for name in ("url", "dump", "file_path", "offset"))
    if any(provenance):
        return "provenance:" + "\x1f".join(provenance)
    return "text:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def clean_finepdfs_record(sample: Sample, context: PipelineContext) -> Sample:
    """Apply the published relaxed LID/minhash/primary-score policy."""
    data = sample.data
    text = data.get("text") if isinstance(data.get("text"), str) else data.get("content")
    if isinstance(text, str):
        data["text"] = text
        sample.text_fields = ("text",)
    lid = _number(data, ("lid_score", "language_score", "language_probability", "language_confidence"))
    minhash = _number(data, ("minhash_count", "minhash_matches", "minhash_overlap"))
    primary = _number(data, ("primary_score", "quality_score", "edu_score", "pdf_quality_score"))
    thresholds = {
        "lid": float(context.config.get("finepdfs_lid_threshold", 0.85)),
        "minhash": float(context.config.get("finepdfs_minhash_max", 2.0)),
        "primary": float(context.config.get("finepdfs_primary_threshold", 0.8)),
    }
    if lid is not None:
        sample.metrics["finepdfs_lid_score"] = lid
        if lid < thresholds["lid"]:
            sample.mark(Decision.DROP, "finepdfs_low_language_score")
    if minhash is not None:
        sample.metrics["finepdfs_minhash_count"] = minhash
        if minhash > thresholds["minhash"]:
            sample.mark(Decision.DROP, "finepdfs_excessive_minhash_overlap")
    if primary is not None:
        sample.metrics["finepdfs_primary_score"] = primary
        if primary < thresholds["primary"]:
            sample.mark(Decision.DROP, "finepdfs_low_primary_score")
    if any(value is None for value in (lid, minhash, primary)):
        sample.flags.add("finepdfs_missing_quality_metric")
        sample.mark(Decision.REVIEW, "finepdfs_missing_quality_metric")
    if isinstance(text, str) and text:
        data["finepdfs_dedup_key"] = _dedup_key(data, text)
    sample.buckets["finepdfs_policy"] = "relaxed_lid085_mh2_p08"
    return sample


FINEPDFS_STEPS = (
    Step("clean_finepdfs_record", clean_finepdfs_record, "Apply FinePDFs language, overlap and quality rules"),
)


__all__ = ["FINEPDFS_STEPS", "clean_finepdfs_record"]
