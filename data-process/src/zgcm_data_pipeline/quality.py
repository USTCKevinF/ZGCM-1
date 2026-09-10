from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .core import PipelineContext
from .models import Decision, Sample


@dataclass(frozen=True, slots=True)
class QualityPolicy:
    """Source-aware score policy shared by classifier and LLM judgements."""

    score_field: str = "quality_score"
    keep_min: float | None = None
    review_min: float | None = None
    hard_error_field: str | None = None
    hard_error_max: float | None = None
    require_score: bool = False

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "QualityPolicy":
        return cls(
            score_field=str(value.get("score_field", "quality_score")),
            keep_min=float(value["keep_min"]) if value.get("keep_min") is not None else None,
            review_min=float(value["review_min"]) if value.get("review_min") is not None else None,
            hard_error_field=str(value["hard_error_field"]) if value.get("hard_error_field") else None,
            hard_error_max=float(value["hard_error_max"]) if value.get("hard_error_max") is not None else None,
            require_score=bool(value.get("require_score", False)),
        )


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value) if isinstance(value, str) and value.strip() else None
    except ValueError:
        return None


def quality_policy_key(sample: Sample) -> str:
    return str(
        sample.data.get("source_key")
        or sample.data.get("source")
        or sample.dataset
        or sample.category.value
    )


def resolve_quality_policy(sample: Sample, config: Mapping[str, object]) -> QualityPolicy | None:
    policies = config.get("quality_policies")
    if not isinstance(policies, Mapping):
        return None
    key = quality_policy_key(sample)
    raw = policies.get(key) or policies.get(sample.category.value) or policies.get("default")
    if isinstance(raw, QualityPolicy):
        return raw
    if isinstance(raw, Mapping):
        return QualityPolicy.from_mapping(raw)
    return None


def apply_quality_policy(sample: Sample, context: PipelineContext) -> Sample:
    """Apply a configured source/category threshold without hard-coding datasets."""
    policy = resolve_quality_policy(sample, context.config)
    if policy is None:
        return sample

    score = _number(sample.data.get(policy.score_field))
    if score is None:
        if policy.require_score:
            sample.flags.add("missing_quality_score")
            sample.mark(Decision.REVIEW, "missing_quality_score")
    else:
        sample.metrics["quality_score"] = score
        if policy.review_min is not None and score < policy.review_min:
            sample.mark(Decision.DROP, "quality_score_below_drop_threshold")
        elif policy.keep_min is not None and score < policy.keep_min:
            sample.mark(Decision.REVIEW, "quality_score_below_keep_threshold")

    if policy.hard_error_field and policy.hard_error_max is not None:
        hard_error = _number(sample.data.get(policy.hard_error_field))
        if hard_error is not None:
            sample.metrics["hard_error_score"] = hard_error
            if hard_error > policy.hard_error_max:
                sample.mark(Decision.DROP, "hard_error_score_exceeded")
    return sample
