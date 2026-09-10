from __future__ import annotations

import argparse
import json
from pathlib import Path

from .adapters import DatasetAdapter
from .core import PipelineContext
from .io import iter_jsonl, write_jsonl
from .models import DataCategory, Stage
from .pipelines import build_pipeline
from .steps.common import release_record
from .common.manifest import build_manifest, write_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a ZGCM-1 category cleaning pipeline")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--stage", choices=[item.value for item in Stage], required=True)
    parser.add_argument("--category", choices=[item.value for item in DataCategory], required=True)
    parser.add_argument("--id-field", default="id")
    parser.add_argument("--text-fields", default="text", help="Comma-separated canonical text fields")
    parser.add_argument("--config", type=Path, help="JSON object with quality, dedup and length policies")
    parser.add_argument("--manifest", type=Path, help="Write a release manifest next to the cleaned JSONL")
    parser.add_argument("--require-license", action="store_true")
    parser.add_argument("--keep-dropped", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    category = DataCategory(args.category)
    adapter = DatasetAdapter(
        dataset=args.dataset,
        stage=Stage(args.stage),
        category=category,
        id_field=args.id_field,
        text_fields=tuple(name.strip() for name in args.text_fields.split(",") if name.strip()),
    )
    config: dict[str, object] = {}
    if args.config:
        loaded = json.loads(args.config.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError("--config must contain one JSON object")
        config.update(loaded)
    if args.require_license:
        config["require_license"] = True
    context = PipelineContext(config=config)
    pipeline = build_pipeline(category)
    samples = adapter.adapt_many(iter_jsonl(args.input))
    cleaned = pipeline.run(samples, context, include_dropped=args.keep_dropped)
    count = write_jsonl(args.output, (release_record(sample) for sample in cleaned))
    summary = {
        "stage": args.stage,
        "category": args.category,
        "dataset": args.dataset,
        "written": count,
        "pipeline": pipeline.step_names(),
        "counters": dict(context.counters),
    }
    if args.manifest:
        # Build the manifest from the serialized output so it exactly matches
        # what a downstream consumer will read.
        rows = (json.loads(line) for line in args.output.read_text(encoding="utf-8").splitlines() if line.strip())
        write_manifest(args.manifest, build_manifest(rows, stage=args.stage, output=args.output.name))
        summary["manifest"] = args.manifest.name
    summary_path = args.summary or args.output.with_suffix(args.output.suffix + ".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
