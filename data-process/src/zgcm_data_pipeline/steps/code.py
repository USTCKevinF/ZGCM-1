from __future__ import annotations

import re

from ..core import PipelineContext, Step
from ..models import Decision, Sample


_GENERATED = re.compile(r"(?i)(?:generated file|do not edit|auto[- ]generated)")
_CODE_SIGNAL = re.compile(r"[{}();]|\b(?:def|class|function|import|package|SELECT|FROM)\b")


def normalize_code_schema(sample: Sample, context: PipelineContext) -> Sample:
    if "content" in sample.data and "text" not in sample.data:
        sample.data["text"] = sample.data["content"]
        sample.text_fields = tuple("text" if name == "content" else name for name in sample.text_fields)
    sample.buckets.setdefault("subtype", str(sample.data.get("subtype") or "code"))
    return sample


def classify_code_language(sample: Sample, context: PipelineContext) -> Sample:
    metadata = sample.data.get("metadata")
    language = sample.data.get("language")
    if not language and isinstance(metadata, dict):
        language = metadata.get("language")
    sample.buckets["language"] = str(language or "unknown")
    return sample


def filter_non_source_assets(sample: Sample, context: PipelineContext) -> Sample:
    path = str(sample.data.get("path") or sample.data.get("file_path") or "").casefold()
    denied = tuple(context.config.get("code_denied_path_parts", ("node_modules/", "vendor/", ".min.js", ".map")))
    if path and any(str(part).casefold() in path for part in denied):
        sample.mark(Decision.DROP, "non_core_or_vendored_file")
    elif _GENERATED.search(sample.text()[:4096]):
        sample.mark(Decision.REVIEW, "possibly_generated_code")
    return sample


def validate_code_signal(sample: Sample, context: PipelineContext) -> Sample:
    text = sample.text()
    if not _CODE_SIGNAL.search(text) and len(text.splitlines()) < 3:
        sample.mark(Decision.REVIEW, "weak_code_signal")
    if "prefix" in sample.data or "middle" in sample.data or "suffix" in sample.data:
        if not all(isinstance(sample.data.get(name), str) and sample.data.get(name) for name in ("prefix", "middle", "suffix")):
            sample.mark(Decision.DROP, "incomplete_fim_triplet")
        sample.buckets["subtype"] = "fim"
    return sample


def bucket_code_task(sample: Sample, context: PipelineContext) -> Sample:
    if sample.data.get("patch") or sample.data.get("diff"):
        sample.buckets["subtype"] = "patch"
    elif sample.data.get("prompt") and sample.data.get("completion"):
        sample.buckets["subtype"] = "code_qa"
    sample.buckets["verified"] = "verified" if sample.data.get("verified") or sample.data.get("tests_passed") else "unverified"
    return sample


CODE_STEPS = (
    Step("normalize_code_schema", normalize_code_schema, "统一源码、问答、FIM 和 patch 字段"),
    Step("classify_code_language", classify_code_language, "识别编程语言"),
    Step("filter_non_source_assets", filter_non_source_assets, "过滤 vendor、生成物和非核心文件"),
    Step("validate_code_signal", validate_code_signal, "检查代码信号和 FIM 完整性"),
    Step("bucket_code_task", bucket_code_task, "按代码任务类型和可验证性分桶"),
)
