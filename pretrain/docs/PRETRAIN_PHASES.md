# General Pretraining

ZGCM-1 is pretrained from scratch on approximately **4.19T tokens** in two
data stages. The recipe below follows Sections 3.1 and A.1 of the ZGCM-1
technical report.

## Data stages

| Stage | Token budget | Data curriculum |
| --- | ---: | --- |
| Stage 1 | Approximately 0.99T | General-language text ordered by lexical complexity, with code and mathematics interleaved independently |
| Stage 2 | Approximately 3.20T | Full data mixture globally shuffled during offline materialization, with increased code and mathematics and an additional specialized reasoning component |

Both stages combine web, academic/OCR, code, mathematics, and LaTeX sources.
Cross-stage deduplication excludes Stage-1 content from Stage-2 selection.
Accepted documents are normalized, materialized as versioned shards, and
tokenized and indexed with the GLM-5.1 tokenizer.

The Stage-1 curriculum orders general-language documents from lower to higher
lexical complexity and removes extreme outliers. Code and mathematics use
independent interleaving because lexical complexity is not a reliable measure
of their reasoning difficulty. Stage 2 is globally shuffled before indexed
dataset creation. Its runtime loader preserves that offline order so checkpoint
resume offsets continue from the same sample sequence.

## Training configuration

| Setting | Value |
| --- | --- |
| Hardware | 192 H100 GPUs |
| Context length | 16,384 tokens |
| Parallelism | TP2 / PP1 / CP2 / DP48 |
| Micro / global batch size | 2 / 768 |
| Activation recomputation | None |
| RoPE base | 5,000,000 |
| Matrix optimizer | Muon, momentum 0.9, spectral scaling, five Newton–Schulz steps |
| Scalar optimizer | Adam |
| Learning rate | Constant `2e-4` |
| Weight decay / gradient clipping | 0.1 / 1.0 |
| Precision | Transformer Engine hybrid FP8 with delayed scaling; BF16 or FP32 for other operations |
| FP8 amax history | 1,024 steps |

The report records approximately 585 model TFLOP/s/GPU for the 16K production
run and estimates an approximately 4.2× improvement in pretraining time-to-loss
over a comparable BF16/AdamW baseline.

## Data and runtime inputs

The [stage configurations](../configs/stages/) declare the runtime, model,
optimizer, dataset, and checkpoint settings. External assets include a Python
environment, tokenizer, Megatron indexed datasets, and checkpoint storage.
See the [Pretrain README](../README.md) for the launch interface and
[source provenance](SOURCE_PROVENANCE.md) for the bundled framework and runtime.
