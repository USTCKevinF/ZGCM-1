"""UltraData message-format cleanup."""

from __future__ import annotations

from ....core import PipelineContext, Step
from ....models import Decision, Sample


FOREIGN_TOKENS = (
    "<|end_header_id|>",
    "<|start_header_id|>",
    "<|model|>",
    "<|messages|>",
    "<|im_start|>",
    "<|im_end|>",
    "<|eot_id|>",
)


def strip_think_tags(text: str) -> str:
    return text.replace("<think>", "").replace("</think>", "").strip()


def repair_visible_assistant_content(text: str) -> tuple[str, bool]:
    if "</think>" not in text and "<think>" not in text:
        return text, False
    if "</think>" in text:
        visible = text.rsplit("</think>", 1)[1].strip()
        if visible:
            return visible, True
    return strip_think_tags(text), True


def clean_ultradata_record(sample: Sample, context: PipelineContext) -> Sample:
    """Remove empty systems, think leaks and foreign template tokens."""
    messages = sample.data.get("messages")
    if not isinstance(messages, list):
        sample.mark(Decision.DROP, "ultradata_bad_messages")
        return sample
    output = []
    for raw in messages:
        if not isinstance(raw, dict):
            sample.mark(Decision.DROP, "ultradata_non_dict_message")
            continue
        message = dict(raw)
        role = message.get("role")
        content = message.get("content")
        for value in (content, message.get("reasoning_content"), message.get("reasoning")):
            if isinstance(value, str) and any(token in value for token in FOREIGN_TOKENS):
                sample.mark(Decision.DROP, "ultradata_foreign_chat_template_token")
                return sample
        if role == "system" and (not isinstance(content, str) or not content.strip()):
            sample.flags.add("ultradata_removed_empty_system")
            continue
        if role == "assistant" and isinstance(content, str):
            repaired, changed = repair_visible_assistant_content(content)
            if changed:
                if not repaired:
                    sample.mark(Decision.DROP, "ultradata_empty_after_think_repair")
                    return sample
                message["content"] = repaired
                sample.flags.add("ultradata_repaired_visible_think_leak")
        for key in ("reasoning_content", "reasoning"):
            value = message.get(key)
            if isinstance(value, str) and ("<think>" in value or "</think>" in value):
                message[key] = strip_think_tags(value)
                sample.flags.add(f"ultradata_stripped_{key}_tag")
        output.append(message)
    if not output:
        sample.mark(Decision.DROP, "ultradata_no_messages_after_clean")
    sample.data["messages"] = output
    sample.buckets["format_profile"] = "ultradata"
    return sample


ULTRADATA_STEPS = (
    Step("clean_ultradata_record", clean_ultradata_record, "Repair UltraData message formatting"),
)


__all__ = [
    "FOREIGN_TOKENS",
    "ULTRADATA_STEPS",
    "clean_ultradata_record",
    "repair_visible_assistant_content",
    "strip_think_tags",
]
