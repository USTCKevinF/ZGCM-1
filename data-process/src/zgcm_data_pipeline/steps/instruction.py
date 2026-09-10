from __future__ import annotations

import re

from ..core import PipelineContext, Step
from ..models import Decision, Sample


_TEMPLATE = re.compile(r"(?i)(?:as an ai language model|i cannot assist with|here is the requested response)")


def normalize_instruction_schema(sample: Sample, context: PipelineContext) -> Sample:
    if "instruction" in sample.data and "prompt" not in sample.data:
        sample.data["prompt"] = sample.data["instruction"]
    if "output" in sample.data and "response" not in sample.data:
        sample.data["response"] = sample.data["output"]
    if isinstance(sample.data.get("prompt"), str) and isinstance(sample.data.get("response"), str):
        sample.text_fields = ("prompt", "response")
    elif isinstance(sample.data.get("messages"), list):
        turns = []
        for message in sample.data["messages"]:
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            if isinstance(content, str) and content:
                turns.append(f"{message.get('role') or 'unknown'}: {content}")
        if turns:
            sample.data["text"] = "\n".join(turns)
            sample.text_fields = ("text",)
    return sample


def validate_instruction_pair(sample: Sample, context: PipelineContext) -> Sample:
    if isinstance(sample.data.get("messages"), list) and sample.data.get("messages"):
        roles = {str(message.get("role") or "") for message in sample.data["messages"] if isinstance(message, dict)}
        if "user" not in roles or "assistant" not in roles:
            sample.mark(Decision.DROP, "incomplete_instruction_messages")
        return sample
    prompt = str(sample.data.get("prompt") or "")
    response = str(sample.data.get("response") or "")
    if not prompt or not response:
        sample.mark(Decision.DROP, "incomplete_instruction_pair")
    elif prompt.casefold() == response.casefold():
        sample.mark(Decision.DROP, "prompt_response_identical")
    return sample


def filter_instruction_templates(sample: Sample, context: PipelineContext) -> Sample:
    if _TEMPLATE.search(str(sample.data.get("response") or "")):
        sample.mark(Decision.REVIEW, "generic_model_template")
    return sample


def bucket_instruction(sample: Sample, context: PipelineContext) -> Sample:
    sample.buckets["task"] = str(sample.data.get("task") or sample.data.get("category") or "general")
    sample.buckets["risk"] = str(sample.data.get("risk_category") or "normal")
    sample.buckets["turns"] = "multi" if isinstance(sample.data.get("messages"), list) and len(sample.data["messages"]) > 2 else "single"
    return sample


INSTRUCTION_STEPS = (
    Step("normalize_instruction_schema", normalize_instruction_schema, "统一 instruction、prompt、response 和 messages"),
    Step("validate_instruction_pair", validate_instruction_pair, "检查指令与回答完整性和退化样本"),
    Step("filter_instruction_templates", filter_instruction_templates, "识别模型套话、拒答模板和系统残留"),
    Step("bucket_instruction", bucket_instruction, "按任务、风险和轮次分桶"),
)
