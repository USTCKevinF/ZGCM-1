from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

try:  # Python 3.11+
    from enum import StrEnum
except ImportError:  # pragma: no cover - exercised only on Python 3.10
    class StrEnum(str, Enum):
        """Small compatibility fallback for Python 3.10."""

        def __str__(self) -> str:
            return self.value
from typing import Any


class Stage(StrEnum):
    PRETRAIN = "pretrain"
    MIDTRAIN = "midtrain"
    POSTTRAIN = "posttrain"
    # Backward-compatible name used by the original archive and existing callers.
    SFT = "sft"


class DataCategory(StrEnum):
    CODE = "code"
    WEB = "web"
    AGENTIC = "agentic"
    INSTRUCTION = "instruction"
    MATH = "math"
    REASONING = "reasoning"
    # Coarse Pretrain source families. They use the shared pipeline only;
    # source-specific parsing lives in stages.pretrain.source_cleaning.
    PDF_OCR = "pdf_ocr"
    GENERAL_TEXT = "general_text"


class Decision(StrEnum):
    KEEP = "keep"
    REVIEW = "review"
    DROP = "drop"


@dataclass(slots=True)
class Sample:
    """Canonical record shared by all category pipelines.

    Dataset adapters map raw fields into ``data`` and declare the fields that
    contain training text. Category steps may add normalized fields, metrics,
    flags and buckets without discarding the original record.
    """

    dataset: str
    source_id: str
    stage: Stage
    category: DataCategory
    data: dict[str, Any]
    text_fields: tuple[str, ...] = ("text",)
    decision: Decision = Decision.KEEP
    reasons: list[str] = field(default_factory=list)
    flags: set[str] = field(default_factory=set)
    buckets: dict[str, str] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)

    def text(self) -> str:
        values: list[str] = []
        for field_name in self.text_fields:
            value = self.data.get(field_name)
            if isinstance(value, str) and value:
                values.append(value)
        return "\n".join(values)

    def mark(self, decision: Decision, reason: str) -> None:
        if decision == Decision.DROP or (
            decision == Decision.REVIEW and self.decision == Decision.KEEP
        ):
            self.decision = decision
        if reason and reason not in self.reasons:
            self.reasons.append(reason)

    def as_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "source_id": self.source_id,
            "stage": self.stage.value,
            "category": self.category.value,
            "decision": self.decision.value,
            "reasons": self.reasons,
            "flags": sorted(self.flags),
            "buckets": self.buckets,
            "metrics": self.metrics,
            "data": self.data,
        }
