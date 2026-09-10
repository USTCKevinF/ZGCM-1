"""Environment-neutral Posttrain (SFT) pipeline entry points."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from ...adapters import DatasetAdapter
from ...common.io import iter_jsonl, write_jsonl
from ...common.manifest import build_manifest, write_manifest
from ...core import Pipeline, PipelineContext
from ...models import DataCategory, Stage
from ...pipelines import build_pipeline as build_category_pipeline
from ...steps.common import release_record
from .datasets import dataset_steps
from .format import POSTTRAIN_FORMAT_STEPS


POSTTRAIN_CATEGORIES = ("instruction", "agentic", "math", "reasoning", "code", "web")


def build_pipeline(category: str, dataset_profile: str | None = None) -> Pipeline:
    """Build a Posttrain pipeline with format normalization before shared checks."""
    key = str(category).strip().lower()
    if key not in POSTTRAIN_CATEGORIES:
        raise ValueError(f"unsupported posttrain category: {category!r}; choose from {POSTTRAIN_CATEGORIES}")
    category_pipeline = build_category_pipeline(DataCategory(key))
    profile_steps = dataset_steps(dataset_profile, key)
    return Pipeline(
        name=f"posttrain_{key}_{dataset_profile or 'generic'}_pipeline",
        steps=profile_steps + POSTTRAIN_FORMAT_STEPS + category_pipeline.steps,
    )


def run_jsonl(
    input_path: Path,
    output_path: Path,
    *,
    dataset: str,
    category: str,
    id_field: str = "id",
    text_fields: tuple[str, ...] = ("text",),
    config: Mapping[str, object] | None = None,
    dataset_profile: str | None = None,
    include_dropped: bool = False,
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Run Posttrain format and quality processing with explicit inputs only."""
    key = str(category).strip().lower()
    adapter = DatasetAdapter(
        dataset=dataset,
        stage=Stage.POSTTRAIN,
        category=DataCategory(key),
        id_field=id_field,
        text_fields=text_fields,
    )
    context = PipelineContext(config=dict(config or {}))
    pipeline = build_pipeline(key, dataset_profile)
    records = (release_record(sample) for sample in pipeline.run(adapter.adapt_many(iter_jsonl(input_path)), context, include_dropped=include_dropped))
    written = write_jsonl(output_path, records)
    summary: dict[str, Any] = {
        "stage": Stage.POSTTRAIN.value,
        "category": key,
        "dataset": dataset,
        "dataset_profile": dataset_profile,
        "written": written,
        "pipeline": pipeline.step_names(),
        "counters": dict(context.counters),
    }
    if manifest_path is not None:
        write_manifest(manifest_path, build_manifest(_read_jsonl(output_path), stage=Stage.POSTTRAIN.value, output=output_path.name))
        summary["manifest"] = manifest_path.name
    return summary


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            yield json.loads(line)


__all__ = ["POSTTRAIN_CATEGORIES", "build_pipeline", "run_jsonl"]
