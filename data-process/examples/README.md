# Example Data and Configuration

These small files demonstrate input fields, profile selection, and cleaning decisions. They are synthetic or reduced examples and are not production training corpora. The directory contains JSONL/JSON inputs only; executable checks live in [`../scripts/README.md`](../scripts/README.md).

| File | Purpose |
|---|---|
| [`instruction.jsonl`](instruction.jsonl) | Generic instruction input with one complete row and one empty answer. |
| [`quality-policies.json`](quality-policies.json) | Web/math quality thresholds and optional near-dedup configuration. License review can be enabled with `--require-license`. |
| [`datasets/gharchive.jsonl`](datasets/gharchive.jsonl) | Pretrain code input with one source file and one `vendor/` file. |
| [`datasets/web_knowledge.jsonl`](datasets/web_knowledge.jsonl) | Midtrain web/knowledge input with quality scores and token counts. |
| [`datasets/dolci_tooluse.jsonl`](datasets/dolci_tooluse.jsonl) | Posttrain legacy function-calling input with tools, calls, and an environment observation. |

Run the examples from the `data-process/` directory with the commands shown in the stage READMEs. Each CLI can write a summary and, with `--manifest`, a manifest. Add `--keep-dropped` to retain dropped rows for inspection.

Default fields are `id` and `text`; use `--id-field` and `--text-fields` when a dataset uses different names. The examples validate record-level behavior only. Production use still requires the target tokenizer, sharding/writer integration, and training-loader checks.
