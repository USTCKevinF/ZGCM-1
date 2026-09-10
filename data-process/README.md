# ZGCM-1 Data Pipeline

This directory contains the public Python package for ZGCM-1 data governance. It is an AI-native governance scaffold: the shared pipeline constrains the input schema, output contract, decision states, and audit fields, while AI creates a separate adapter or profile for each dataset. Dataset-specific rules stay in the profile; common validation, quality, deduplication, length, and manifest behavior stays reusable.

The package accepts JSONL records and produces cleaned JSONL with decisions and a manifest. It runs in a local Python environment and does not require a cluster, private mount, or vendor runtime. Large-scale extraction, sharding, tokenization, and indexed-dataset writing are supplied by the target training environment.

## Layout

```text
data-process/
├── pyproject.toml                         # Package metadata and CLI entry points
├── README.md
├── examples/                              # Minimal JSONL/JSON inputs
│   ├── README.md
│   ├── instruction.jsonl                  # Generic instruction example
│   ├── quality-policies.json              # Quality/dedup configuration example
│   └── datasets/                          # Dataset-profile examples
├── scripts/
│   ├── README.md
│   └── smoke_test.py                      # End-to-end smoke test for four CLIs
├── src/zgcm_data_pipeline/
│   ├── common/                            # Shared schema, quality, dedup, length, manifest helpers
│   ├── steps/                             # Code/web/instruction/etc. category steps
│   ├── stages/
│   │   ├── pretrain/                      # Code, PDF/OCR, and general text
│   │   ├── midtrain/                      # Category cleaning and length mixing
│   │   └── posttrain/                     # SFT messages, tools, and packing
│   ├── adapters.py                        # Raw fields → canonical Sample adapter
│   ├── core.py                            # Pipeline, Step, and PipelineContext
│   ├── models.py                          # Stage, category, decision, and Sample models
│   └── training.py                        # Tokenization/packing/loss-mask interfaces
├── tests/
└── docs/                                  # Public category inventory and pipeline diagram
```

Example fields and expected behavior are documented in [`examples/README.md`](examples/README.md). The local helper script is documented in [`scripts/README.md`](scripts/README.md).

## Stage entry points

| Stage | CLI module | Scope | Stage README |
|---|---|---|---|
| Pretrain | `zgcm_data_pipeline.stages.pretrain.cli` | `code`, `pdf_ocr`, and `general_text` | [README](src/zgcm_data_pipeline/stages/pretrain/README.md) |
| Midtrain | `zgcm_data_pipeline.stages.midtrain.cli` | `code`, `web`, `instruction`, `agentic`, `math`, and `reasoning` | [README](src/zgcm_data_pipeline/stages/midtrain/README.md) |
| Posttrain | `zgcm_data_pipeline.stages.posttrain.cli` | Message, reasoning, tool-call, and packing preparation | [README](src/zgcm_data_pipeline/stages/posttrain/README.md) |

The package also provides the generic `zgcm_data_pipeline` category CLI. All four entry points accept `--input`, `--output`, `--dataset`, `--config`, `--manifest`, `--summary`, and `--keep-dropped`; stage entry points additionally validate `--source`, `--category`, and `--dataset-profile`.

## Processing contract

```text
JSONL input
  → DatasetAdapter field mapping
  → schema and text normalization
  → integrity, provenance, license, and sensitive-content checks
  → AI-generated dataset profile
  → stage/category shared steps
  → quality decision (keep/review/drop)
  → exact deduplication and optional near deduplication
  → token estimate and length bucket
  → cleaned JSONL, summary, and manifest
```

Each serialized record contains at least:

```json
{
  "dataset": "example",
  "source_id": "row-1",
  "stage": "midtrain",
  "category": "instruction",
  "decision": "keep",
  "reasons": [],
  "flags": [],
  "buckets": {},
  "metrics": {},
  "data": {},
  "record_sha256": "..."
}
```

`data` keeps the source record and normalized fields. `reasons` explains review or drop decisions; `flags` records audit annotations; `buckets` and `metrics` support sampling and reporting. Dropped records are omitted by default but remain counted in the summary; use `--keep-dropped` for inspection. The manifest is rebuilt from serialized output so its record count matches what a downstream loader reads.

## AI-native governance model

The shared pipeline provides stable guardrails rather than one fixed rule set for every dataset. AI inspects a dataset's fields, noise patterns, and task objective, then produces an independent adapter/profile that repairs fields, applies source-specific filters, and records dataset-specific metrics. The profile plugs into the stage pipeline through `DatasetAdapter`, so different datasets can retain their own rules while sharing the same decisions, audit fields, and downstream interface. The modules under `stages/*/datasets/` and the files under `examples/datasets/` are representative implementations, not a complete catalog of all dataset cleaners.

## Installation and quick start

Python 3.10 or newer is supported. The core package uses only the standard library. Install `zstandard` separately when reading `.zst` input.
Creating an editable installation may require access to a Python package index
to obtain the build requirement declared in `pyproject.toml`. When that is not
available, the examples and tests can be run directly with `PYTHONPATH=src`.

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e .
```

Run the generic instruction example:

```bash
OUTPUT_DIR="${OUTPUT_DIR:-outputs}"
mkdir -p "${OUTPUT_DIR}"
PYTHONPATH=src python3 -m zgcm_data_pipeline \
  --input examples/instruction.jsonl \
  --output "${OUTPUT_DIR}/instruction.cleaned.jsonl" \
  --dataset example_instruction \
  --stage midtrain \
  --category instruction \
  --text-fields instruction,output \
  --config examples/quality-policies.json \
  --manifest "${OUTPUT_DIR}/instruction.manifest.json"
```

Stage profile commands are shown in each stage README. After installation, the console scripts `zgcm-clean`, `zgcm-pretrain-clean`, `zgcm-midtrain-clean`, and `zgcm-posttrain-clean` are also available.

## Configuration and length mixing

`examples/quality-policies.json` shows category quality thresholds and optional near deduplication. License review can be enabled with `--require-license` or `require_license` in a JSON configuration. Dataset profiles accept their own namespaced settings, such as `finepdfs_*`, `olmocr_*`, `web_knowledge_*`, and `dolci_think_*`.

Without an injected training tokenizer, shared steps use a reproducible token estimate. Production integration should provide the target tokenizer, chat template, indexed writer, and loader validation.

Midtrain `buckets.py` assigns `B16`, `B64`, and `B256` base buckets. `mix.py` uses stable hashing to construct disjoint prefix mixes:

```text
mix16  ← B16
mix64  ← B16 + B64, excluding IDs already selected for mix16
mix256 ← B16 + B64 + B256, excluding IDs already selected for mix16/mix64
```

The mix function returns selected IDs; quota policy, materialization, and indexed-dataset writing remain training-environment integrations.

Sampling helpers use `zgcm` as the default seed; `prefix_sample_mix` uses `zgcm-midtrain`. Set `seed` explicitly and record it with the dataset configuration to reproduce the same partitioning, ordering, and sample selection across package versions.

## Local verification

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
python3 -m compileall -q src
python3 scripts/smoke_test.py
```

The tests cover shared category pipelines, dataset profiles, length mixing, message conversion, and representative CLI inputs. Before training, validate token boundaries, loss masks, checksums, and full-corpus distributions in the target environment.

Public category inventory and cleaning diagrams are in [`docs/dataset_category_inventory.md`](docs/dataset_category_inventory.md) and [`docs/category_cleaning_pipelines.md`](docs/category_cleaning_pipelines.md).
