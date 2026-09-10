"""Environment-neutral Posttrain message and tool-format normalization."""

from __future__ import annotations

import re
from typing import Any

from ...core import PipelineContext, Step
from ...models import Sample


_THINK = re.compile(r"(?is)<think>\s*(.*?)\s*</think>")
_ROLE_ALIASES = {"human": "user", "gpt": "assistant", "model": "assistant", "function": "tool"}


def split_think_content(content: str) -> tuple[str | None, str]:
    """Split one assistant message into hidden reasoning and visible content."""
    match = _THINK.search(content)
    if not match:
        return None, content.strip()
    reasoning = match.group(1).strip()
    visible = _THINK.sub("", content).strip()
    return reasoning or None, visible


def normalize_messages(sample: Sample, context: PipelineContext) -> Sample:
    """Normalize role aliases, content types and optional think fields in-place."""
    messages = sample.data.get("messages")
    if not isinstance(messages, list):
        return sample
    normalized: list[dict[str, Any]] = []
    for raw in messages:
        if not isinstance(raw, dict):
            continue
        message = dict(raw)
        role = str(message.get("role") or "").casefold()
        message["role"] = _ROLE_ALIASES.get(role, role or "unknown")
        content = message.get("content")
        if content is not None and not isinstance(content, str):
            message["content"] = str(content)
        if message["role"] == "assistant" and isinstance(message.get("content"), str):
            reasoning, visible = split_think_content(message["content"])
            message["content"] = visible
            if reasoning and not message.get("reasoning_content"):
                message["reasoning_content"] = reasoning
        normalized.append(message)
    sample.data["messages"] = normalized
    sample.metrics["turns"] = float(len(normalized))
    return sample


def normalize_tool_calls(sample: Sample, context: PipelineContext) -> Sample:
    """Normalize lightweight tool-call fields without prescribing a vendor schema."""
    messages = sample.data.get("messages")
    if not isinstance(messages, list):
        return sample
    for message in messages:
        if not isinstance(message, dict):
            continue
        calls = message.get("tool_calls")
        if not isinstance(calls, list):
            continue
        normalized_calls = []
        for call in calls:
            if not isinstance(call, dict):
                continue
            item = dict(call)
            if "function" in item and isinstance(item["function"], dict):
                function = dict(item["function"])
                item.setdefault("name", function.get("name"))
                item.setdefault("arguments", function.get("arguments"))
            if item.get("arguments") is not None and not isinstance(item["arguments"], str):
                item["arguments"] = str(item["arguments"])
            normalized_calls.append(item)
        message["tool_calls"] = normalized_calls
    return sample


POSTTRAIN_FORMAT_STEPS = (
    Step("normalize_messages", normalize_messages, "Normalize roles, content and think fields"),
    Step("normalize_tool_calls", normalize_tool_calls, "Normalize tool-call metadata"),
)


__all__ = ["POSTTRAIN_FORMAT_STEPS", "normalize_messages", "normalize_tool_calls", "split_think_content"]
