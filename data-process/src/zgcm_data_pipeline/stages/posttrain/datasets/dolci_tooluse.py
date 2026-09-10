"""Dolci legacy tool-use to native messages/tools conversion."""

from __future__ import annotations

import ast
import json
import re
from typing import Any

from ....core import PipelineContext, Step
from ....models import Decision, Sample


LEGACY_TOOL_SYSTEM_RE = re.compile(
    r"^\s*You are a helpful function-calling AI assistant\.\s*"
    r"You are provided with function signatures within <functions></functions> XML tags\.\s*"
    r"You may call one or more functions to assist with the user query\.\s*"
    r"Output any function calls within <function_calls></function_calls> XML tags\.\s*"
    r"Don't make assumptions about what values to plug into functions\.\s*",
    re.IGNORECASE,
)


def load_functions(value: Any) -> list[dict[str, Any]]:
    if not value:
        return []
    raw = json.loads(value) if isinstance(value, str) else value
    if not isinstance(raw, list):
        raise ValueError("functions field is not a list")
    tools: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "function" and isinstance(item.get("function"), dict):
            tools.append({"type": "function", "function": dict(item["function"])})
        elif item.get("name"):
            tools.append({"type": "function", "function": dict(item)})
    return tools


def strip_legacy_system(content: Any) -> str:
    text = content if isinstance(content, str) else ""
    text = LEGACY_TOOL_SYSTEM_RE.sub("", text).strip()
    return "" if "<functions>" in text or "<function_calls>" in text else text


def split_calls(text: str) -> list[str]:
    calls: list[str] = []
    start: int | None = None
    depth = 0
    quote: str | None = None
    escaped = False
    for index, char in enumerate(text):
        if start is None:
            if char.isspace():
                continue
            start = index
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue
        if char in {"'", '"'}:
            quote = char
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0 and start is not None:
                calls.append(text[start : index + 1].strip())
                start = None
        elif char == "\n" and depth == 0 and start is not None:
            piece = text[start:index].strip()
            if piece:
                calls.append(piece)
            start = None
    if start is not None and text[start:].strip():
        calls.append(text[start:].strip())
    return calls


def _sanitize(value: Any) -> Any:
    if value is Ellipsis:
        return "..."
    if isinstance(value, complex):
        return str(value)
    if isinstance(value, (set, tuple, list)):
        return [_sanitize(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _sanitize(item) for key, item in value.items()}
    return value


def _literal(node: ast.AST) -> Any:
    try:
        return _sanitize(ast.literal_eval(node))
    except Exception:
        return ast.unparse(node) if hasattr(ast, "unparse") else str(node)


def parse_arguments(text: str) -> dict[str, Any]:
    if not text.strip():
        return {}
    expression = ast.parse(f"f({text})", mode="eval").body
    if not isinstance(expression, ast.Call):
        raise ValueError("arguments did not parse as a function call")
    result = {f"arg{index}": _literal(value) for index, value in enumerate(expression.args)}
    for keyword in expression.keywords:
        result[keyword.arg or "kwargs"] = _literal(keyword.value)
    return result


def parse_function_calls(value: Any) -> list[dict[str, Any]]:
    if not value or (isinstance(value, str) and value.strip().casefold() in {"null", "none"}):
        return []
    if isinstance(value, list):
        return value
    output: list[dict[str, Any]] = []
    for call in split_calls(str(value).strip()):
        left, right = call.find("("), call.rfind(")")
        if left <= 0 or right <= left:
            raise ValueError(f"bad function call syntax: {call[:120]}")
        output.append(
            {
                "type": "function",
                "function": {
                    "name": call[:left].strip(),
                    "arguments": parse_arguments(call[left + 1 : right]),
                },
            }
        )
    return output


def convert_dolci_tooluse_record(sample: Sample, context: PipelineContext) -> Sample:
    """Convert one legacy Dolci Tool Use record to native top-level tools."""
    messages = sample.data.get("messages")
    if not isinstance(messages, list) or not messages:
        sample.mark(Decision.DROP, "dolci_tooluse_empty_messages")
        return sample
    tools: list[dict[str, Any]] = []
    output: list[dict[str, Any]] = []
    try:
        for raw in messages:
            if not isinstance(raw, dict):
                continue
            role = raw.get("role")
            if role == "system":
                tools.extend(load_functions(raw.get("functions")))
                content = strip_legacy_system(raw.get("content"))
                if content:
                    output.append({"role": "system", "content": content})
            elif role == "environment":
                output.append({"role": "tool", "content": raw.get("content") or ""})
            elif role == "assistant" and raw.get("function_calls"):
                calls = parse_function_calls(raw.get("function_calls"))
                if not calls:
                    raise ValueError("assistant function_calls is empty")
                output.append({"role": "assistant", "content": "", "tool_calls": calls})
            elif role in {"user", "assistant", "tool"}:
                output.append({"role": role, "content": raw.get("content") or ""})
    except (SyntaxError, TypeError, ValueError, json.JSONDecodeError):
        sample.mark(Decision.DROP, "dolci_tooluse_conversion_failed")
        return sample
    sample.data["messages"] = output
    sample.data["tools"] = tools
    sample.data["tool_schema_format"] = "native_tools_v1"
    sample.buckets["tool_schema"] = "native"
    if not tools:
        sample.mark(Decision.REVIEW, "dolci_tooluse_missing_tools")
    return sample


def validate_dolci_tooluse_record(sample: Sample, context: PipelineContext) -> Sample:
    messages = sample.data.get("messages")
    if not isinstance(messages, list):
        return sample
    roles: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            sample.mark(Decision.DROP, "dolci_tooluse_bad_message")
            continue
        role = str(message.get("role") or "")
        roles.append(role)
        if role not in {"system", "user", "assistant", "tool"}:
            sample.mark(Decision.DROP, "dolci_tooluse_bad_role")
        if role == "assistant" and not message.get("content") and not message.get("tool_calls"):
            sample.mark(Decision.DROP, "dolci_tooluse_empty_assistant")
        if role == "assistant" and message.get("tool_calls"):
            for call in message["tool_calls"]:
                function = call.get("function") if isinstance(call, dict) else None
                if not isinstance(function, dict) or not function.get("name") or not isinstance(function.get("arguments"), dict):
                    sample.mark(Decision.DROP, "dolci_tooluse_bad_tool_call")
    for index, role in enumerate(roles):
        if role == "system" and index != 0:
            sample.mark(Decision.DROP, "dolci_tooluse_system_not_first")
        if role == "assistant" and (index == 0 or roles[index - 1] not in {"user", "tool"}):
            sample.mark(Decision.DROP, "dolci_tooluse_invalid_assistant_order")
        if role == "tool" and (index == 0 or roles[index - 1] not in {"assistant", "tool"}):
            sample.mark(Decision.DROP, "dolci_tooluse_invalid_tool_order")
    return sample


DOLCI_TOOLUSE_STEPS = (
    Step("convert_dolci_tooluse_record", convert_dolci_tooluse_record, "Convert Dolci legacy function calls"),
    Step("validate_dolci_tooluse_record", validate_dolci_tooluse_record, "Validate converted Dolci tool messages"),
)


__all__ = [
    "DOLCI_TOOLUSE_STEPS",
    "convert_dolci_tooluse_record",
    "load_functions",
    "parse_arguments",
    "parse_function_calls",
    "split_calls",
    "strip_legacy_system",
    "validate_dolci_tooluse_record",
]
