"""Web/Knowledge classifier-score and token-length bucketing."""

from __future__ import annotations

from typing import Any

from ....core import PipelineContext, Step
from ....models import Decision, Sample


def quality_band(score: float) -> str:
    if score >= 4.0:
        return "ge4"
    if score >= 3.0:
        return "ge3"
    return "lt3"


def length_bucket(token_count: int) -> str | None:
    if token_count <= 0:
        return None
    if token_count <= 16_384:
        return "le_16k"
    if token_count <= 65_536:
        return "gt_16k_le_64k"
    if token_count <= 262_144:
        return "gt_64k_le_256k"
    return "gt_256k"


def _value(data: dict[str, Any], key: str) -> Any:
    if key in data:
        return data[key]
    metadata = data.get("meta") if isinstance(data.get("meta"), dict) else data.get("metadata")
    return metadata.get(key) if isinstance(metadata, dict) else None


def clean_web_knowledge_record(sample: Sample, context: PipelineContext) -> Sample:
    """Attach the archived ge4/ge3/lt3 and 16K/64K/256K split labels."""
    score = _value(sample.data, "quality_mean")
    try:
        score_value = float(score)
    except (TypeError, ValueError):
        score_value = -1.0
        sample.data["quality_missing"] = True
        sample.flags.add("web_knowledge_missing_quality_score")
        sample.mark(Decision.REVIEW, "web_knowledge_missing_quality_score")
    else:
        sample.data["quality_missing"] = False
    band = quality_band(score_value)
    sample.data["quality_mean"] = score_value
    sample.data["quality_band"] = band
    sample.metrics["web_knowledge_quality_mean"] = score_value
    sample.buckets["classifier_quality"] = band

    token_count = _value(sample.data, "token_count")
    try:
        token_value = int(token_count)
    except (TypeError, ValueError):
        token_value = 0
    bucket = length_bucket(token_value)
    if bucket is None:
        sample.flags.add("web_knowledge_missing_token_count")
        sample.mark(Decision.REVIEW, "web_knowledge_missing_token_count")
    else:
        sample.metrics["training_token_count"] = float(token_value)
        sample.buckets["source_length"] = bucket
    minimum_band = str(context.config.get("web_knowledge_min_quality_band", "lt3"))
    order = {"lt3": 0, "ge3": 1, "ge4": 2}
    if minimum_band not in order:
        raise ValueError("web_knowledge_min_quality_band must be one of lt3, ge3, ge4")
    if order[band] < order[minimum_band]:
        sample.mark(Decision.DROP, "web_knowledge_below_quality_band")
    return sample


WEB_KNOWLEDGE_STEPS = (
    Step(
        "clean_web_knowledge_record",
        clean_web_knowledge_record,
        "Apply Web/Knowledge classifier quality and length buckets",
    ),
)


__all__ = ["WEB_KNOWLEDGE_STEPS", "clean_web_knowledge_record", "length_bucket", "quality_band"]
