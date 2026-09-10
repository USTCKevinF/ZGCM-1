"""Dolci-Think eight-source message cleaning."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

from ....core import PipelineContext, Step
from ....models import Decision, Sample


SOURCE_LABELS = {
    "saumyamalik/OpenThoughts3-full-filtered-math-decontam-v2": "openthoughts_math",
    "saumyamalik/correct-python-sft-187k-x16-thoughts-filtered-decontam-v2": "dolci_python_code",
    "allenai/persona-precise-if-r1-final-content-filtered-chinese-filtered": "persona_precise_if",
    "saumyamalik/if_qwq_reasoning_verified_filtered_decontam-v2": "qwq_verified_if",
    "allenai/nemotron-post-training-dataset-subset-ngram-filtered-no-tool-calls": "nemotron_code_no_tool",
    "allenai/SYNTHETIC-2-SFT-cn-fltrd-final-ngram-filtered-chinese-filtered": "synthetic2_verified",
    "saumyamalik/OpenThoughts3-full-filtered-science-decontam-v2": "openthoughts_science",
    "saumyamalik/OpenThoughts3-full-filtered-code-subsampled-decontam-v2": "openthoughts_code",
}

ABILITY_BY_LABEL = {
    "openthoughts_math": "math_reasoning",
    "dolci_python_code": "code_reasoning",
    "persona_precise_if": "precise_if",
    "qwq_verified_if": "precise_if",
    "nemotron_code_no_tool": "code_reasoning",
    "synthetic2_verified": "code_basic_reasoning",
    "openthoughts_science": "science_reasoning",
    "openthoughts_code": "code_reasoning",
}

THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
TEMPLATE_TOKEN_RE = re.compile(
    r"<\|(?:end_header_id|start_header_id|eot_id|messages|user|assistant|system|model|observation|endoftext)[^>]*\|>"
    r"|\[gMASK\]|<\|im_start\|>|<\|im_end\|>",
    re.IGNORECASE,
)
WORD_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")


def rough_tokens(text: str) -> int:
    return len(WORD_RE.findall(text or ""))


def parse_assistant(content: str) -> tuple[str | None, str | None, str | None]:
    match = THINK_RE.search(content or "")
    if not match:
        return None, None, "missing_complete_think"
    reasoning = match.group(1).strip()
    answer = content[match.end() :].strip()
    if not reasoning:
        return reasoning, answer, "empty_think"
    if not answer:
        return reasoning, answer, "empty_visible_answer"
    if THINK_RE.search(content[match.end() :]):
        return reasoning, answer, "multiple_think_blocks"
    return reasoning, answer, None


def has_repetition(text: str) -> bool:
    words = WORD_RE.findall((text or "").casefold())
    if len(words) < 120:
        return False
    grams = [" ".join(words[index : index + 5]) for index in range(len(words) - 4)]
    counts = Counter(grams)
    repeated = sum(count - 1 for count in counts.values() if count > 1)
    return repeated / max(1, len(grams)) > 0.20


def clean_dolci_think_record(sample: Sample, context: PipelineContext) -> Sample:
    """Split rendered think blocks and apply the archived strict format checks."""
    data = sample.data
    source_name = data.get("dataset_source")
    label = SOURCE_LABELS.get(str(source_name))
    if label is None:
        sample.mark(Decision.DROP, "dolci_think_not_target_source")
        return sample
    messages = data.get("messages")
    if not isinstance(messages, list) or not messages:
        sample.mark(Decision.DROP, "dolci_think_bad_messages")
        return sample
    cleaned: list[dict[str, Any]] = []
    rendered_tokens = 0
    strict = bool(context.config.get("dolci_think_strict", True))
    for raw in messages:
        if not isinstance(raw, dict):
            sample.mark(Decision.DROP, "dolci_think_bad_message_object")
            continue
        role = str(raw.get("role") or "")
        content = raw.get("content") if isinstance(raw.get("content"), str) else ""
        if role == "assistant":
            reasoning, answer, error = parse_assistant(content)
            if error:
                sample.mark(Decision.DROP, f"dolci_think_{error}")
            reasoning = reasoning or ""
            answer = answer or ""
            if TEMPLATE_TOKEN_RE.search(content):
                sample.mark(Decision.DROP if strict else Decision.REVIEW, "dolci_think_template_token")
            if has_repetition(reasoning) or has_repetition(answer):
                sample.mark(Decision.DROP if strict else Decision.REVIEW, "dolci_think_repetition")
            cleaned.append({"role": "assistant", "content": answer, "reasoning_content": reasoning})
            rendered_tokens += rough_tokens(reasoning) + rough_tokens(answer) + 4
        else:
            cleaned.append({"role": role, "content": content})
            rendered_tokens += rough_tokens(content)
    maximum = int(context.config.get("dolci_think_max_rough_tokens", 32_768))
    if rendered_tokens > maximum:
        sample.mark(Decision.DROP, "dolci_think_over_length")
    elif rendered_tokens > int(context.config.get("dolci_think_review_rough_tokens", 16_000)):
        sample.mark(Decision.REVIEW, "dolci_think_long_record")
    data["messages"] = cleaned
    data["source"] = label
    data["source_key"] = label
    data["ability"] = ABILITY_BY_LABEL[label]
    data["think_mode"] = "think"
    metadata = data.setdefault("metadata", {})
    if isinstance(metadata, dict):
        metadata["rough_render_tokens"] = rendered_tokens
        metadata["source_label"] = label
    sample.metrics["dolci_think_rough_tokens"] = float(rendered_tokens)
    sample.buckets["dolci_source"] = label
    return sample


DOLCI_THINK_STEPS = (
    Step("clean_dolci_think_record", clean_dolci_think_record, "Clean Dolci rendered think messages"),
)


__all__ = [
    "DOLCI_THINK_STEPS",
    "SOURCE_LABELS",
    "clean_dolci_think_record",
    "has_repetition",
    "parse_assistant",
]
