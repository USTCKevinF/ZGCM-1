"""Command-line entry point for Posttrain (SFT) cleaning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .datasets import DATASET_PROFILES
from .pipeline import POSTTRAIN_CATEGORIES, run_jsonl


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run an environment-neutral ZGCM-1 Posttrain pipeline")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--category", choices=sorted(POSTTRAIN_CATEGORIES), required=True)
    parser.add_argument("--dataset-profile", choices=sorted(DATASET_PROFILES), help="Optional dataset-specific cleaning profile")
    parser.add_argument("--id-field", default="id")
    parser.add_argument("--text-fields", default="text", help="Comma-separated text fields")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--require-license", action="store_true", help="Mark records without license metadata for review")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--keep-dropped", action="store_true")
    parser.add_argument("--summary", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config: dict[str, object] = {}
    if args.config:
        loaded = json.loads(args.config.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError("--config must contain one JSON object")
        config = loaded
    if args.require_license:
        config["require_license"] = True
    summary = run_jsonl(
        args.input,
        args.output,
        dataset=args.dataset,
        category=args.category,
        dataset_profile=args.dataset_profile,
        id_field=args.id_field,
        text_fields=tuple(item.strip() for item in args.text_fields.split(",") if item.strip()),
        config=config,
        include_dropped=args.keep_dropped,
        manifest_path=args.manifest,
    )
    summary_path = args.summary or args.output.with_suffix(args.output.suffix + ".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
