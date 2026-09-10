from __future__ import annotations

import re

from ..core import PipelineContext, Step
from ..models import Decision, Sample


_CONTROL_ARTIFACT = re.compile(r"(?is)<\|(?:system|assistant|user|endoftext)[^>]*\|>|\[system prompt\]|api[_ ]error")
_REASON_SIGNAL = re.compile(r"(?i)(?:because|therefore|first,|step \d+|we need to|由于|因此|首先|步骤)")


def normalize_reasoning_schema(sample: Sample, context: PipelineContext) -> Sample:
    if "problem" not in sample.data and "prompt" in sample.data:
        sample.data["problem"] = sample.data["prompt"]
    if "reasoning" not in sample.data and "response" in sample.data:
        sample.data["reasoning"] = sample.data["response"]
    if isinstance(sample.data.get("problem"), str) and isinstance(sample.data.get("reasoning"), str):
        fields = ["problem", "reasoning"]
        if isinstance(sample.data.get("final_answer"), str):
            fields.append("final_answer")
        sample.text_fields = tuple(fields)
    return sample


def detect_control_artifacts(sample: Sample, context: PipelineContext) -> Sample:
    if _CONTROL_ARTIFACT.search(sample.text()):
        sample.mark(Decision.REVIEW, "control_or_prompt_artifact")
    return sample


def validate_reasoning_signal(sample: Sample, context: PipelineContext) -> Sample:
    reasoning = str(sample.data.get("reasoning") or "")
    if not reasoning:
        sample.mark(Decision.DROP, "missing_reasoning")
    elif len(reasoning) < 64 or not _REASON_SIGNAL.search(reasoning):
        sample.mark(Decision.REVIEW, "weak_reasoning_signal")
    if not sample.data.get("final_answer") and not sample.data.get("answer"):
        sample.mark(Decision.REVIEW, "missing_final_answer")
    return sample


def bucket_reasoning(sample: Sample, context: PipelineContext) -> Sample:
    sample.buckets["origin"] = "teacher_rewrite" if sample.data.get("teacher_model") else str(sample.data.get("origin") or "raw")
    sample.buckets["domain"] = str(sample.data.get("domain") or sample.data.get("source_group") or "general")
    sample.buckets["verified"] = "verified" if sample.data.get("verified") else "unverified"
    upstream = sample.data.get("source_row_hash")
    if upstream:
        sample.data.setdefault("lineage_hash", str(upstream))
    return sample


REASONING_STEPS = (
    Step("normalize_reasoning_schema", normalize_reasoning_schema, "统一 problem、reasoning 和 final_answer"),
    Step("detect_control_artifacts", detect_control_artifacts, "检查系统提示、特殊 token 和 API 残留"),
    Step("validate_reasoning_signal", validate_reasoning_signal, "检查推理完整性和最终答案"),
    Step("bucket_reasoning", bucket_reasoning, "按来源、领域、teacher/raw 和验证状态分桶"),
)

