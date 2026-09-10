# Midtrain Stage

The Midtrain stage handles data that needs category-level quality control and length-aware quotas after basic normalization. It supports code, web, instruction, agentic, math, and reasoning categories. Dataset-specific field repair belongs in a profile; reusable schema, quality, deduplication, and bucketing stay in shared steps.

The shared Midtrain pipeline is the governance layer for AI-assisted cleaning. AI can create an independent profile for each web, math, or agentic dataset and still reuse the same decisions, audit fields, and length-mix interface. The three modules under `datasets/` demonstrate this pattern.

## Files and scripts

| File | Description |
|---|---|
| [`cli.py`](cli.py) | Stage CLI `zgcm-midtrain-clean`; handles arguments, configuration, and JSONL/manifest output. |
| [`pipeline.py`](pipeline.py) | Validates the category, prepends the dataset profile, appends the category pipeline, and exposes `run_jsonl()`. |
| [`buckets.py`](buckets.py) | Assigns base length buckets using 16K, 64K, and 256K token boundaries. |
| [`mix.py`](mix.py) | Uses stable hashing for disjoint `mix16`/`mix64`/`mix256` prefix sampling. |
| [`datasets/web_knowledge.py`](datasets/web_knowledge.py) | Reads web/knowledge quality scores and token counts, then assigns quality bands and length buckets. |
| [`datasets/nemotron_math.py`](datasets/nemotron_math.py) | Filters Math records by `(source_file, row_index)` drop keys and hard-error scores. |
| [`datasets/agent_coding.py`](datasets/agent_coding.py) | Normalizes pre-rendered agent-coding traces and records rough length, dedup, lineage, completion, and priority metadata. |
| [`datasets/__init__.py`](datasets/__init__.py) | Registers profiles and their allowed categories. |

Category steps live in [`steps/`](../../steps/): Code handles source signals and languages; Web handles QA and boilerplate; Instruction handles prompt/response structure; Agentic handles trajectory closure; Math handles problem/answer signals; Reasoning handles reasoning signals and control artifacts.

## Dataset profiles

| `--dataset-profile` | Category | Rules |
|---|---|---|
| `web_knowledge` | `web` | Maps scores to `ge4`/`ge3`/`lt3`; assigns `le_16k`, `gt_16k_le_64k`, `gt_64k_le_256k`, and `gt_256k` source-length buckets; missing scores or lengths are marked for review. |
| `nemotron_math` | `math` | Drops configured source-row keys or hard-error scores at or above `4.0` by default. |
| `agent_coding` | `agentic` | Records character count, rough tokens, length bucket, text hashes, trace keys, completion status, and same-repository priority. |

Profile/category compatibility is strict. For example, `web_knowledge` cannot be used with `math`. Without a profile, any supported category can use the generic category pipeline.

## Length buckets and prefix mixing

`assign_length_bucket(token_count)` uses these default base buckets:

| Token count | Bucket |
|---:|---|
| `<= 16,384` | `B16` |
| `16,384 < n <= 65,536` | `B64` |
| `65,536 < n <= 262,144` | `B256` |
| `> 262,144` | `gt_256` |

`prefix_sample_mix(base_buckets, final_targets, seed=...)` uses the following candidate pools:

```text
mix16  <- B16
mix64  <- B16 + B64, excluding IDs selected for mix16
mix256 <- B16 + B64 + B256, excluding IDs selected for mix16/mix64
```

Short samples remain eligible for longer mixes, while each selected ID appears in only one final mix. The function returns IDs; quota policy, tokenization, materialization, and indexed-dataset writing are training-environment integrations.

## CLI example

Run from the `data-process/` directory:

```bash
OUTPUT_DIR="${OUTPUT_DIR:-outputs}"
mkdir -p "${OUTPUT_DIR}"
PYTHONPATH=src python3 -m zgcm_data_pipeline.stages.midtrain.cli \
  --input examples/datasets/web_knowledge.jsonl \
  --output "${OUTPUT_DIR}/web_knowledge.cleaned.jsonl" \
  --dataset web_knowledge \
  --category web \
  --dataset-profile web_knowledge \
  --manifest "${OUTPUT_DIR}/web_knowledge.manifest.json"
```

Use a JSON configuration to require a higher minimum quality band:

```json
{"web_knowledge_min_quality_band": "ge3"}
```

Generic options are documented in the package [README](../../../../README.md).

## Examples in this stage

[`examples/datasets/web_knowledge.jsonl`](../../../../examples/datasets/web_knowledge.jsonl) is the Midtrain-specific input. Its two records have quality scores `4.2` and `2.4`; the default `lt3` minimum keeps both and assigns quality/length buckets, while `ge3` or `ge4` drops the lower-scoring record.

[`examples/instruction.jsonl`](../../../../examples/instruction.jsonl) is also runnable through the generic Midtrain instruction pipeline: its complete row is kept and its empty `output` row is dropped. `nemotron_math` and `agent_coding` currently have profiles but no dedicated input examples. See [`examples/README.md`](../../../../examples/README.md) for the complete list.

## Output and limits

Outputs follow the shared record contract, and summaries include per-step counters. Token counts can be approximate without a real tokenizer; production integration should provide source quotas and validate mix sizes, disjointness, checksums, and training-loader readability.
