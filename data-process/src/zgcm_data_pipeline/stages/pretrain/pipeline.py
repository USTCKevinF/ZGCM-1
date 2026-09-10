"""Environment-neutral Pretrain pipeline entry points."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from ...adapters import DatasetAdapter
from ...core import Pipeline, PipelineContext
from ...io import iter_jsonl, write_jsonl
from ...models import DataCategory, Stage
from ...steps.common import COMMON_PREFIX, COMMON_SUFFIX, release_record
from ...common.manifest import build_manifest, write_manifest
from .datasets import dataset_steps
from .source_cleaning import PRETRAIN_CODE_STEPS, PRETRAIN_PDF_STEPS, PRETRAIN_TEXT_STEPS


SOURCE_CATEGORIES = {
    "code": DataCategory.CODE,
    "pdf_ocr": DataCategory.PDF_OCR,
    "general_text": DataCategory.GENERAL_TEXT,
}


def build_pipeline(source: str, dataset_profile: str | None = None) -> Pipeline:
    """Build a Pretrain pipeline for one of the three supported source families."""
    key = str(source).strip().lower()
    if key not in SOURCE_CATEGORIES:
        raise ValueError(f"unsupported pretrain source: {source!r}; choose from {sorted(SOURCE_CATEGORIES)}")
    if key == "code":
        source_steps = PRETRAIN_CODE_STEPS
    elif key == "pdf_ocr":
        source_steps = PRETRAIN_PDF_STEPS
    else:
        source_steps = PRETRAIN_TEXT_STEPS
    return Pipeline(
        name=f"pretrain_{key}_{dataset_profile or 'generic'}_pipeline",
        steps=dataset_steps(dataset_profile, key) + source_steps + COMMON_PREFIX + COMMON_SUFFIX,
    )


def run_jsonl(
    input_path: Path,
    output_path: Path,
    *,
    dataset: str,
    source: str,
    id_field: str = "id",
    text_fields: tuple[str, ...] = ("text",),
    config: Mapping[str, object] | None = None,
    dataset_profile: str | None = None,
    include_dropped: bool = False,
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Run a Pretrain source pipeline without assuming any local environment."""
    key = str(source).strip().lower()
    category = SOURCE_CATEGORIES.get(key)
    if category is None:
        raise ValueError(f"unsupported pretrain source: {source!r}")
    adapter = DatasetAdapter(
        dataset=dataset,
        stage=Stage.PRETRAIN,
        category=category,
        id_field=id_field,
        text_fields=text_fields,
    )
    context = PipelineContext(config=dict(config or {}))
    pipeline = build_pipeline(key, dataset_profile)
    records = (release_record(sample) for sample in pipeline.run(adapter.adapt_many(iter_jsonl(input_path)), context, include_dropped=include_dropped))
    written = write_jsonl(output_path, records)
    summary: dict[str, Any] = {
        "stage": Stage.PRETRAIN.value,
        "source": key,
        "dataset": dataset,
        "dataset_profile": dataset_profile,
        "written": written,
        "pipeline": pipeline.step_names(),
        "counters": dict(context.counters),
    }
    if manifest_path is not None:
        write_manifest(manifest_path, build_manifest(_read_jsonl(output_path), stage=Stage.PRETRAIN.value, output=output_path.name))
        summary["manifest"] = manifest_path.name
    return summary


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            yield json.loads(line)


__all__ = ["SOURCE_CATEGORIES", "build_pipeline", "run_jsonl"]
