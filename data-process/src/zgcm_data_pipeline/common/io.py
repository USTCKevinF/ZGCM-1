"""Path-agnostic JSONL and optional Zstandard I/O."""

from ..io import iter_jsonl, open_text, write_jsonl

__all__ = ["iter_jsonl", "open_text", "write_jsonl"]
