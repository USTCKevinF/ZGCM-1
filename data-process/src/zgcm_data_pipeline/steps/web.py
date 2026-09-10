from __future__ import annotations

import re

from ..core import PipelineContext, Step
from ..models import Decision, Sample


_BOILERPLATE = re.compile(r"(?i)(cookie policy|accept all cookies|privacy policy|sign in|subscribe now)")
_UNKNOWN = re.compile(r"(?i)^\s*(?:unknown|n/?a|i don'?t know|无法回答|不知道)\s*[.!。！]?$", re.M)


def normalize_web_schema(sample: Sample, context: PipelineContext) -> Sample:
    question = sample.data.get("question")
    answer = sample.data.get("answer")
    if isinstance(question, str) and isinstance(answer, str):
        sample.text_fields = ("question", "answer")
        sample.buckets["subtype"] = "web_qa"
    else:
        sample.buckets.setdefault("subtype", "web_text")
    return sample


def filter_web_boilerplate(sample: Sample, context: PipelineContext) -> Sample:
    text = sample.text()
    matches = len(_BOILERPLATE.findall(text))
    if matches >= int(context.config.get("web_boilerplate_limit", 4)):
        sample.mark(Decision.REVIEW, "web_boilerplate")
    lines = [line for line in text.splitlines() if line.strip()]
    if lines and len(set(lines)) / len(lines) < 0.45:
        sample.mark(Decision.DROP, "repetitive_web_text")
    return sample


def validate_web_qa(sample: Sample, context: PipelineContext) -> Sample:
    if sample.buckets.get("subtype") != "web_qa":
        return sample
    question = str(sample.data.get("question") or "")
    answer = str(sample.data.get("answer") or "")
    if len(question) < 4 or not answer:
        sample.mark(Decision.DROP, "incomplete_qa_pair")
    elif _UNKNOWN.match(answer):
        sample.mark(Decision.DROP, "unknown_answer")
    if sample.data.get("grounded") is False:
        sample.mark(Decision.REVIEW, "ungrounded_answer")
    return sample


def bucket_web(sample: Sample, context: PipelineContext) -> Sample:
    sample.buckets["topic"] = str(sample.data.get("topic_top1") or sample.data.get("topic") or "unknown")
    score = sample.data.get("quality_score")
    if isinstance(score, (int, float)):
        sample.buckets["model_quality"] = "high" if score >= 0.8 else "medium" if score >= 0.5 else "low"
    return sample


WEB_STEPS = (
    Step("normalize_web_schema", normalize_web_schema, "统一网页正文和网页 QA 字段"),
    Step("filter_web_boilerplate", filter_web_boilerplate, "过滤模板、导航和重复网页文本"),
    Step("validate_web_qa", validate_web_qa, "检查问答完整性、未知答案和 grounded 信号"),
    Step("bucket_web", bucket_web, "按网页形态、主题和质量分桶"),
)

