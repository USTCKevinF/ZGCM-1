from __future__ import annotations

from typing import Any

from ..core import PipelineContext, Step
from ..models import Decision, Sample


def _messages(sample: Sample) -> list[dict[str, Any]]:
    value = sample.data.get("messages") or sample.data.get("trajectory") or []
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def normalize_agentic_schema(sample: Sample, context: PipelineContext) -> Sample:
    messages = _messages(sample)
    if messages and not isinstance(sample.data.get("text"), str):
        turns = []
        for message in messages:
            content = message.get("content")
            if isinstance(content, str) and content:
                turns.append(f"{message.get('role') or 'unknown'}: {content}")
        if turns:
            sample.data["text"] = "\n".join(turns)
            sample.text_fields = ("text",)
    if not messages and isinstance(sample.data.get("text"), str) and not sample.data.get("pre_rendered_agent_trace"):
        sample.mark(Decision.REVIEW, "unstructured_agent_trace")
    sample.metrics["turns"] = float(len(messages))
    sample.buckets["subtype"] = str(sample.data.get("sample_type") or "trajectory")
    return sample


def validate_role_order(sample: Sample, context: PipelineContext) -> Sample:
    roles = [str(item.get("role") or "") for item in _messages(sample)]
    if not roles:
        return sample
    if roles[0] not in {"system", "user"}:
        sample.mark(Decision.REVIEW, "invalid_first_role")
    if any(not role for role in roles):
        sample.mark(Decision.DROP, "missing_message_role")
    return sample


def validate_tool_call_pairs(sample: Sample, context: PipelineContext) -> Sample:
    if sample.data.get("pre_rendered_agent_trace"):
        return sample
    pending = 0
    tool_calls = 0
    observations = 0
    for message in _messages(sample):
        role = str(message.get("role") or "")
        has_call = bool(message.get("tool_calls")) or role in {"assistant_tool", "tool_call"}
        if has_call:
            pending += 1
            tool_calls += 1
        if role in {"tool", "tool_response", "observation"}:
            observations += 1
            pending = max(0, pending - 1)
    sample.metrics["tool_calls"] = float(tool_calls)
    sample.metrics["tool_observations"] = float(observations)
    if tool_calls == 0:
        sample.mark(Decision.REVIEW, "no_tool_call")
    elif pending:
        sample.mark(Decision.REVIEW, "unpaired_tool_call")
    return sample


def detect_trajectory_outcome(sample: Sample, context: PipelineContext) -> Sample:
    text = sample.text().casefold()
    failure_markers = ("exit due to", "permission denied", "tool error", "timed out")
    success_markers = ("submitted", "tests passed", "task completed", "success")
    failed = any(marker in text for marker in failure_markers)
    verified = bool(sample.data.get("verified") or sample.data.get("tests_passed") or any(marker in text for marker in success_markers))
    sample.buckets["outcome"] = "verified" if verified else "failed" if failed else "weak"
    if failed and not verified:
        sample.mark(Decision.REVIEW, "failed_or_environment_trajectory")
    return sample


def bucket_agentic(sample: Sample, context: PipelineContext) -> Sample:
    tool_names = sample.data.get("tool_names") or sample.data.get("target_tools") or []
    if isinstance(tool_names, list) and tool_names:
        sample.buckets["tool_family"] = ",".join(sorted({str(name) for name in tool_names}))
    else:
        sample.buckets.setdefault("tool_family", "unknown")
    return sample


AGENTIC_STEPS = (
    Step("normalize_agentic_schema", normalize_agentic_schema, "统一 messages、tools、actions 和 observations"),
    Step("validate_role_order", validate_role_order, "检查对话角色和轮次顺序"),
    Step("validate_tool_call_pairs", validate_tool_call_pairs, "检查工具调用与观察结果闭环"),
    Step("detect_trajectory_outcome", detect_trajectory_outcome, "识别完成、测试、提交和异常退出信号"),
    Step("bucket_agentic", bucket_agentic, "按轨迹类型、工具和结果分桶"),
)
