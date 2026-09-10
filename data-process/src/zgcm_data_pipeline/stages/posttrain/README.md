# Posttrain Stage (SFT)

The Posttrain stage converts supervised fine-tuning data into a common `messages`/`tools` structure. It handles hidden reasoning content, tool-call trajectories, and the boundary between record cleaning and training-time packing. The public package implements record-level format and quality logic; the target training environment supplies the chat template, tokenizer, special-token IDs, indexed writer, and loader.

The shared Posttrain pipeline provides format and training-interface constraints. AI can create an independent profile for each SFT, tool-use, or reasoning dataset. `dolci_think.py`, `dolci_tooluse.py`, and `ultradata.py` demonstrate profiles for reasoning content, tool calls, and message repair.

## Files and scripts

| File | Description |
|---|---|
| [`cli.py`](cli.py) | Stage CLI `zgcm-posttrain-clean`; reads JSONL and writes cleaned records, a summary, and an optional manifest. |
| [`pipeline.py`](pipeline.py) | Runs the dataset profile, message/tool format steps, and shared category steps in order. |
| [`format.py`](format.py) | Normalizes role aliases and content types, splits `<think>...</think>`, and normalizes lightweight tool-call fields. |
| [`packing.py`](packing.py) | Exposes tokenized-sample validation, target loss masks, and packing primitives without binding a tokenizer. |
| [`datasets/dolci_think.py`](datasets/dolci_think.py) | Selects supported Dolci sources, splits reasoning/content, and checks template tokens, repetition, and rough length. |
| [`datasets/dolci_tooluse.py`](datasets/dolci_tooluse.py) | Converts legacy `functions`, textual `function_calls`, and `environment` records into native tools, assistant `tool_calls`, and `tool` messages; validates role order. |
| [`datasets/ultradata.py`](datasets/ultradata.py) | Removes empty system messages, repairs visible assistant think leaks, strips reasoning tags, and rejects foreign chat-template tokens. |
| [`datasets/__init__.py`](datasets/__init__.py) | Registers profiles and their allowed categories. |

## Dataset profiles

| `--dataset-profile` | Allowed category | Rules |
|---|---|---|
| `dolci_think` | `instruction` | Accepts supported source labels, moves complete assistant `<think>` blocks to `reasoning_content`, and applies strict/review checks for template tokens, repetition, and length. |
| `dolci_tooluse` | `agentic` | Parses function signatures, function-call arguments, and environment observations into a native tool schema; validates role order and call arguments. |
| `ultradata` | `instruction` | Removes empty systems, repairs visible think remnants, strips reasoning tags, and rejects foreign template tokens while retaining repair flags. |

Profiles run before shared message normalization. Without a profile, choose `instruction`, `agentic`, `math`, `reasoning`, `code`, or `web` for generic category processing. Profile/category compatibility is strict.

## Format and packing behavior

The shared formatter:

- maps `human`, `gpt`, `model`, and `function` aliases to `user`, `assistant`, and `tool`;
- converts non-string content to strings and records the number of turns;
- moves the first assistant `<think>...</think>` block into `reasoning_content`;
- normalizes `name` and `arguments` inside lightweight `tool_calls` without imposing a vendor schema.

`packing.py` accepts already-tokenized samples. Special tokens, assistant spans, target loss masks, and `cu_seqlens` must be configured and validated with the training chat template.

## CLI example

Run from the `data-process/` directory:

```bash
OUTPUT_DIR="${OUTPUT_DIR:-outputs}"
mkdir -p "${OUTPUT_DIR}"
PYTHONPATH=src python3 -m zgcm_data_pipeline.stages.posttrain.cli \
  --input examples/datasets/dolci_tooluse.jsonl \
  --output "${OUTPUT_DIR}/dolci_tooluse.cleaned.jsonl" \
  --dataset dolci_tooluse \
  --category agentic \
  --dataset-profile dolci_tooluse \
  --manifest "${OUTPUT_DIR}/dolci_tooluse.manifest.json"
```

Use `--keep-dropped` to retain dropped rows for inspection. Dolci Think strictness and rough-length limits can be changed with `dolci_think_strict`, `dolci_think_review_rough_tokens`, and `dolci_think_max_rough_tokens` in `--config`.

## Examples in this stage

[`examples/datasets/dolci_tooluse.jsonl`](../../../../examples/datasets/dolci_tooluse.jsonl) is the only Posttrain-specific input. It declares a `weather` function in a system message, calls it as `weather(city='Beijing')`, includes an environment result, and ends with an assistant answer. The profile moves the function into top-level `tools`, the call into assistant `tool_calls`, and the environment result into a `tool` message.

`dolci_think` and `ultradata` currently have profile implementations but no dedicated JSONL examples. `instruction.jsonl` is a generic instruction example for the Midtrain CLI and is not a Posttrain profile input. See [`examples/README.md`](../../../../examples/README.md) for the complete list.

## Output and limits

Outputs contain the shared `decision`, `reasons`, `flags`, `buckets`, `metrics`, and `record_sha256` fields. Dropped rows are omitted by default but counted in the summary. Before SFT training, validate rendered token boundaries, assistant target masks, packing isolation, and indexed-file readability with the target loader.
