#!/usr/bin/env python3
"""Run the small public examples through every supported CLI.

This intentionally uses only the standard library so it can be run from a
fresh checkout before installing the package.  It catches drift between the
README commands, CLI argument parsers, and the release manifest contract.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(module: str, input_path: str, output_path: Path, *extra: str) -> dict[str, object]:
    manifest = output_path.with_suffix(".manifest.json")
    command = [
        sys.executable,
        "-m",
        module,
        "--input",
        str(ROOT / input_path),
        "--output",
        str(output_path),
        "--manifest",
        str(manifest),
        *extra,
    ]
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    subprocess.run(command, cwd=ROOT, env=env, check=True)
    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    manifest_data = json.loads(manifest.read_text(encoding="utf-8"))
    if manifest_data["records"] != len(rows):
        raise AssertionError(f"manifest/output count mismatch for {module}")
    if any(row.get("decision") not in {"keep", "review", "drop"} for row in rows):
        raise AssertionError(f"invalid decision in {module} output")
    return {"module": module, "records": len(rows), "manifest": manifest.name}


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="zgcm-smoke-") as directory:
        out = Path(directory)
        results = [
            run(
                "zgcm_data_pipeline",
                "examples/instruction.jsonl",
                out / "generic.jsonl",
                "--dataset",
                "example_instruction",
                "--stage",
                "midtrain",
                "--category",
                "instruction",
                "--text-fields",
                "instruction,output",
            ),
            run(
                "zgcm_data_pipeline.stages.pretrain.cli",
                "examples/datasets/gharchive.jsonl",
                out / "pretrain.jsonl",
                "--dataset",
                "gharchive",
                "--source",
                "code",
                "--dataset-profile",
                "gharchive",
            ),
            run(
                "zgcm_data_pipeline.stages.midtrain.cli",
                "examples/datasets/web_knowledge.jsonl",
                out / "midtrain.jsonl",
                "--dataset",
                "web_knowledge",
                "--category",
                "web",
                "--dataset-profile",
                "web_knowledge",
            ),
            run(
                "zgcm_data_pipeline.stages.posttrain.cli",
                "examples/datasets/dolci_tooluse.jsonl",
                out / "posttrain.jsonl",
                "--dataset",
                "dolci_tooluse",
                "--category",
                "agentic",
                "--dataset-profile",
                "dolci_tooluse",
            ),
        ]
    print(json.dumps({"status": "ok", "checks": results}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
