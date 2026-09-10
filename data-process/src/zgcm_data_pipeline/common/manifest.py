"""Portable release manifest and checksum helpers."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


def file_sha256(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return a streaming SHA256 for one local file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(records: Iterable[Mapping[str, Any]], *, stage: str, output: str | None = None) -> dict[str, Any]:
    """Build a small JSON-serializable manifest without embedding local paths."""
    counts: dict[str, int] = {"keep": 0, "review": 0, "drop": 0}
    total = 0
    for record in records:
        total += 1
        decision = str(record.get("decision", "keep"))
        counts[decision] = counts.get(decision, 0) + 1
    manifest: dict[str, Any] = {"stage": stage, "records": total, "decisions": counts}
    if output is not None:
        manifest["output"] = output
    return manifest


def write_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    """Write a deterministic, UTF-8 manifest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(manifest), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


__all__ = ["build_manifest", "file_sha256", "write_manifest"]
