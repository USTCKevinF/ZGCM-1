from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from typing import Any

from ..core import PipelineContext, Step
from ..dedup import NearDeduper
from ..models import Decision, Sample
from ..quality import apply_quality_policy


_WHITESPACE = re.compile(r"[\t\f\v ]+")
_BLANK_LINES = re.compile(r"\n{3,}")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_SECRET = re.compile(
    r"(?i)(?:api[_-]?key|secret|password|access[_-]?token)\s*[:=]\s*[\"']?[A-Za-z0-9_\-/.]{12,}"
)


def stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def normalize_text_value(value: str) -> str:
    value = unicodedata.normalize("NFKC", value.replace("\r\n", "\n").replace("\r", "\n"))
    value = _CONTROL.sub("", value)
    value = "\n".join(_WHITESPACE.sub(" ", line).strip() for line in value.splitlines())
    return _BLANK_LINES.sub("\n\n", value).strip()


def normalize_schema_text(sample: Sample, context: PipelineContext) -> Sample:
    for field_name in sample.text_fields:
        value = sample.data.get(field_name)
        if isinstance(value, str):
            sample.data[field_name] = normalize_text_value(value)
    return sample


def validate_basic_integrity(sample: Sample, context: PipelineContext) -> Sample:
    text = sample.text()
    if not text:
        sample.mark(Decision.DROP, "empty_training_text")
        return sample
    minimum = int(context.config.get("min_chars", 16))
    maximum = int(context.config.get("max_chars", 2_000_000))
    if len(text) < minimum:
        sample.mark(Decision.DROP, "too_short")
    elif len(text) > maximum:
        sample.mark(Decision.REVIEW, "too_long")
    return sample


def scan_sensitive_content(sample: Sample, context: PipelineContext) -> Sample:
    text = sample.text()
    if _SECRET.search(text):
        sample.flags.add("possible_secret")
        sample.mark(Decision.REVIEW, "possible_secret")
    terms = context.config.get("sensitive_terms", ())
    if isinstance(terms, (list, tuple, set)):
        lowered = text.casefold()
        if any(str(term).casefold() in lowered for term in terms if str(term)):
            sample.flags.add("sensitive_term")
            sample.mark(Decision.REVIEW, "sensitive_term")
    return sample


def validate_provenance(sample: Sample, context: PipelineContext) -> Sample:
    required = bool(context.config.get("require_license", False))
    metadata = sample.data.get("metadata")
    license_name = sample.data.get("license")
    if not license_name and isinstance(metadata, dict):
        license_name = metadata.get("license")
    if required and not license_name:
        sample.flags.add("missing_license")
        sample.mark(Decision.REVIEW, "missing_license")
    if not sample.data.get("source"):
        sample.flags.add("missing_source")
        sample.mark(Decision.REVIEW, "missing_source")
    return sample


def exact_deduplicate(sample: Sample, context: PipelineContext) -> Sample:
    normalized = normalize_text_value(sample.text()).casefold()
    digest = stable_hash(normalized)
    sample.data.setdefault("text_sha256", digest)
    if digest in context.seen_hashes:
        sample.mark(Decision.DROP, "exact_duplicate")
    else:
        context.seen_hashes.add(digest)
    return sample


def near_deduplicate(sample: Sample, context: PipelineContext) -> Sample:
    """Optionally apply reusable SimHash near-deduplication.

    It is disabled by default because thresholds must be calibrated per
    category. Enable it with ``near_dedup=True`` in ``PipelineContext.config``.
    """
    if not bool(context.config.get("near_dedup", False)):
        return sample
    if context.near_dedup_state is None:
        context.near_dedup_state = NearDeduper(
            threshold=int(context.config.get("near_dedup_threshold", 3)),
            bands=int(context.config.get("near_dedup_bands", 4)),
        )
    deduper = context.near_dedup_state
    if not isinstance(deduper, NearDeduper):
        raise TypeError("PipelineContext.near_dedup_state must be a NearDeduper")
    if deduper.check_and_add(sample.text(), ngram=int(context.config.get("near_dedup_ngram", 5))):
        sample.mark(Decision.DROP, "near_duplicate")
    return sample


def estimate_tokens_and_length_bucket(sample: Sample, context: PipelineContext) -> Sample:
    text = sample.text()
    estimator = context.config.get("token_estimator")
    if callable(estimator):
        tokens = int(estimator(text))
    else:
        tokens = max(1, (len(text) + 3) // 4)
    tokens = max(1, tokens)
    sample.metrics["estimated_tokens"] = float(tokens)
    boundaries = context.config.get("length_buckets", (2048, 8192, 32768, 65536))
    if isinstance(boundaries, (list, tuple)):
        try:
            boundaries = tuple(sorted({int(value) for value in boundaries if int(value) > 0}))
        except (TypeError, ValueError):
            boundaries = ()
    else:
        boundaries = ()
    if not boundaries:
        boundaries = (2048, 8192, 32768, 65536)
    label = f"gt_{boundaries[-1]}"
    lower = 0
    for upper in boundaries:
        upper = int(upper)
        if tokens <= upper:
            label = f"{lower + 1}_{upper}"
            break
        lower = upper
    sample.buckets["length"] = label
    return sample


def finalize_quality_bucket(sample: Sample, context: PipelineContext) -> Sample:
    if sample.decision == Decision.KEEP:
        bucket = "main"
    elif sample.decision == Decision.REVIEW:
        bucket = "review"
    else:
        bucket = "drop"
    sample.buckets.setdefault("quality", bucket)
    return sample


def release_record(sample: Sample) -> dict[str, Any]:
    """Stable serialized form used by materializers and manifests."""
    result = sample.as_dict()
    result["record_sha256"] = stable_hash(
        json.dumps(result["data"], ensure_ascii=False, sort_keys=True, default=str)
    )
    return result


COMMON_PREFIX = (
    Step("normalize_schema_text", normalize_schema_text, "统一 Unicode、换行与文本字段"),
    Step("validate_basic_integrity", validate_basic_integrity, "检查空值、损坏和长度边界"),
    Step("validate_provenance", validate_provenance, "检查来源和许可证元数据"),
    Step("scan_sensitive_content", scan_sensitive_content, "扫描敏感词、密钥与隐私风险"),
)

COMMON_SUFFIX = (
    Step("apply_quality_policy", apply_quality_policy, "按数据源或类别应用程序化、分类器或 LLM 质量阈值"),
    Step("exact_deduplicate", exact_deduplicate, "对规范化训练文本做稳定哈希去重"),
    Step("near_deduplicate", near_deduplicate, "可选 SimHash 近似去重，阈值按类别配置"),
    Step("estimate_tokens_and_length_bucket", estimate_tokens_and_length_bucket, "统计 token 并分长度桶"),
    Step("finalize_quality_bucket", finalize_quality_bucket, "统一输出 main/review/drop 质量桶"),
)
