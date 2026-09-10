# Mid-Training Curriculum

ZGCM-1 extends its context through **16K → 64K → 256K** continued pretraining.
Sections 3.2 and A.1 of the technical report describe the data curriculum,
optimization settings, and long-context qualification experiments.

## Three-stage data curriculum

The training schedule samples **600.51B tokens** from a deduplicated
2.86T-token candidate pool.

| Stage | Maximum context | Token budget | Global batch size |
| --- | ---: | ---: | ---: |
| 16K | 16,384 | 180B | 768 |
| 64K | 65,536 | 240B | 192 |
| 256K | 262,144 | 180.51B | 48 |

Each stage processes approximately 12.6M tokens per optimizer step. Later
stages retain shorter examples while introducing longer sequences:

| Training stage | Up to 16K | Above 16K, up to 64K | Above 64K |
| --- | ---: | ---: | ---: |
| 64K | 180.89B | 59.11B | — |
| 256K | 127.81B | 21.72B | 30.98B |

The mixture covers code, mathematics, knowledge, reasoning, instruction, and
agentic data. Interaction traces are reformulated as Markov Decision Process
state-action transitions to provide supervision for individual decisions.
All stages use full-sequence causal language modeling over raw context.

## Optimization

The runnable stage configurations use cosine decay from `2e-5` to `2e-6`.
The 256K stage uses a RoPE base of 10M and full activation recomputation. The
training framework combines hybrid attention, Muon optimization, and FP8
computation.

## Long-context qualification

The report separately compares two routes to 256K on 192 H100 GPUs:

| Route | Context | Token budget | Parallelism | Micro / global batch | Recompute | RoPE base |
| --- | ---: | ---: | --- | --- | --- | ---: |
| Direct | 256K | 30B | TP2 / PP2 / CP4 / DP12 | 1 / 48 | Full | 10M |
| Staged, step 1 | 64K | 10B | TP2 / PP1 / CP4 / DP24 | 1 / 192 | Selective | 5M |
| Staged, step 2 | 256K | 20B | TP2 / PP2 / CP4 / DP12 | 1 / 48 | Full | 10M |

These are qualification experiments within the long-context study. The staged
route reaches a final loss of 1.16 versus 1.19 for the direct route at comparable
throughput. The main data curriculum is the 600.51B-token schedule above.

## Runtime inputs

Runtime settings are supplied through the [stage configurations](../configs/stages/).
Training consumes Megatron indexed datasets and initializes from pretrained
weights. See the [Midtrain README](../README.md) for the launch interface and
[source provenance](SOURCE_PROVENANCE.md) for the framework and runtime.
