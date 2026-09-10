"""Agent-coding trace normalization and source-priority metadata."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any

from ....core import PipelineContext, Step
from ....models import Decision, Sample


DEDUP_PRIORITY = {
    "agent_coding_same_repo_multiissue_pack": 0,
    "agent_coding_trace64k_packed_full": 1,
    "agent_coding_nonzerohero_singletrace": 2,
}


def rough_length_bucket(chars: int) -> str:
    tokens = max(1, int(chars / 3.6))
    if tokens <= 16_384:
        return "le16k"
    if tokens <= 65_536:
        return "gt16k_le64k"
    if tokens <= 262_144:
        return "gt64k_le256k"
    return "gt256k"


def source_keys(data: Mapping[str, Any]) -> set[str]:
    metadata = data.get("metadata") if isinstance(data.get("metadata"), Mapping) else {}
    result: set[str] = set()
    for name in ("source_id", "source_instance_id"):
        value = metadata.get(name)
        if isinstance(value, str) and value:
            result.add(value)
    for name in ("source_ids", "source_instance_ids"):
        value = metadata.get(name)
        if isinstance(value, list):
            result.update(str(item) for item in value if item)
    return result


def sampled_text_for_neardup(text: str, span: int = 12_000) -> str:
    if len(text) <= span * 3:
        return text
    middle = max(0, len(text) // 2 - span // 2)
    return text[:span] + "\n" + text[middle : middle + span] + "\n" + text[-span:]


def clean_agent_coding_record(sample: Sample, context: PipelineContext) -> Sample:
    """Normalize the pre-rendered AgentTrove-style records used by Midtrain."""
    data = sample.data
    text = data.get("text")
    if not isinstance(text, str) or not text.strip():
        sample.mark(Decision.DROP, "agent_coding_missing_trace_text")
        return sample
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    subtype = str(data.get("sample_type") or "agent_coding_trace64k_packed_full")
    chars = len(text)
    rough_tokens = max(1, int(chars / 3.6))
    digest = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()
    data["text"] = text
    data["text_chars"] = chars
    data["rough_tokens_chars_div_3_6"] = rough_tokens
    data["length_bucket"] = rough_length_bucket(chars)
    data["dedup_text_sha256"] = digest
    data["pre_rendered_agent_trace"] = True
    data.setdefault("training_tags", ["agentic", "coding", "agent_trace"])
    data.setdefault("completion_status", "completed_success")
    data["dedup_priority"] = DEDUP_PRIORITY.get(subtype, max(DEDUP_PRIORITY.values()) + 1)
    data["source_trace_keys"] = sorted(source_keys(data))
    data["near_dedup_sample_sha256"] = hashlib.sha256(
        sampled_text_for_neardup(text).encode("utf-8", errors="replace")
    ).hexdigest()
    sample.text_fields = ("text",)
    sample.buckets["subtype"] = subtype
    sample.buckets["source_length"] = data["length_bucket"]
    if str(data.get("completion_status")).casefold() not in {"completed_success", "success", "passed"}:
        sample.mark(Decision.REVIEW, "agent_coding_unverified_completion")
    return sample


AGENT_CODING_STEPS = (
    Step("clean_agent_coding_record", clean_agent_coding_record, "Normalize Agent Coding trace records"),
)


__all__ = [
    "AGENT_CODING_STEPS",
    "clean_agent_coding_record",
    "rough_length_bucket",
    "sampled_text_for_neardup",
    "source_keys",
]
