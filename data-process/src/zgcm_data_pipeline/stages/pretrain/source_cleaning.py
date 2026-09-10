"""Pretrain-only source-family normalization.

The functions intentionally contain no cluster paths or credentials. Dataset
specific extraction can be supplied through ``DatasetAdapter.field_map`` and
hooks while these compact steps keep the public pipeline reproducible.
"""

from __future__ import annotations

from ...core import PipelineContext, Step
from ...models import Decision, Sample
from ...steps.code import (
    bucket_code_task,
    classify_code_language,
    filter_non_source_assets,
    normalize_code_schema,
    validate_code_signal,
)


def normalize_pdf_ocr(sample: Sample, context: PipelineContext) -> Sample:
    """Map common PDF/OCR fields to the canonical ``text`` field."""
    if "text" not in sample.data:
        for key in ("content", "ocr_text", "document"):
            if isinstance(sample.data.get(key), str):
                sample.data["text"] = sample.data[key]
                break
    sample.text_fields = ("text",)
    sample.buckets["source_family"] = "pdf_ocr"
    if not sample.data.get("text"):
        sample.mark(Decision.DROP, "missing_extracted_text")
    return sample


def filter_pdf_ocr_quality(sample: Sample, context: PipelineContext) -> Sample:
    """Apply portable OCR quality checks; thresholds are configuration-only."""
    text = sample.text()
    min_chars = int(context.config.get("pdf_min_chars", 32))
    if len(text) < min_chars:
        sample.mark(Decision.REVIEW, "short_ocr_text")
    if sample.data.get("ocr_quality") is not None:
        try:
            score = float(sample.data["ocr_quality"])
        except (TypeError, ValueError):
            score = None
        if score is not None and score < float(context.config.get("pdf_min_ocr_quality", 0.0)):
            sample.mark(Decision.REVIEW, "low_ocr_quality")
    return sample


def normalize_general_text(sample: Sample, context: PipelineContext) -> Sample:
    """Map generic text/document fields to canonical text."""
    if "text" not in sample.data:
        for key in ("content", "document", "body", "raw_text"):
            if isinstance(sample.data.get(key), str):
                sample.data["text"] = sample.data[key]
                break
    sample.text_fields = ("text",)
    sample.buckets["source_family"] = "general_text"
    if not sample.data.get("text"):
        sample.mark(Decision.DROP, "missing_text")
    return sample


PRETRAIN_CODE_STEPS = (
    Step("normalize_code_schema", normalize_code_schema, "Map code content fields"),
    Step("classify_code_language", classify_code_language, "Classify code language"),
    Step("filter_non_source_assets", filter_non_source_assets, "Filter generated and vendored files"),
    Step("validate_code_signal", validate_code_signal, "Validate code signal"),
    Step("bucket_code_task", bucket_code_task, "Assign code subtype"),
)

PRETRAIN_PDF_STEPS = (
    Step("normalize_pdf_ocr", normalize_pdf_ocr, "Normalize PDF/OCR text"),
    Step("filter_pdf_ocr_quality", filter_pdf_ocr_quality, "Check OCR quality and length"),
)

PRETRAIN_TEXT_STEPS = (
    Step("normalize_general_text", normalize_general_text, "Normalize general text"),
)


__all__ = [
    "PRETRAIN_CODE_STEPS",
    "PRETRAIN_PDF_STEPS",
    "PRETRAIN_TEXT_STEPS",
    "filter_pdf_ocr_quality",
    "normalize_general_text",
    "normalize_pdf_ocr",
]
