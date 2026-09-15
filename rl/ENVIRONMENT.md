# Environment

The mixed-RL experiments in Section 4.2 of the technical report use a policy
model and tokenizer, a mixture of mathematics/code/general prompts, and
domain-specific reward evaluation.

The training framework is [AReaL](https://github.com/areal-project/AReaL)
**v2.0.0** (commit `fee938e`). Install AReaL from source at that commit and use
the dependency set below, which is the exact stack the released runs were
verified on (Python 3.12):

| Package | Version |
| --- | --- |
| torch | 2.11.0+cu129 |
| vllm | 0.25.1+cu129 |
| transformers | 5.14.1 |
| ray | 2.56.0 |
| math-verify | 0.8.0 |
| flashinfer-python | 0.6.13 |
| antlr4-python3-runtime | 4.13.2 |

Experiment inputs include:

- Policy weights and the associated tokenizer (HF checkpoint; loading ZGCM-1
  requires `trust_remote_code`).
- Prompts selected using initial-policy difficulty estimates.
- Answer verifiers for mathematics and correctness-based general tasks.
- Executable tests for code rewards.
- Instruction-following criteria for the corresponding general tasks.

The final math GRPO stage needs only the mathematics verifier shipped in
`rewards/`; the code/general reward handlers of the internal pipeline are not
required to reproduce it.

Rollouts allow up to 65,536 generated tokens within a 98,304-token total
prompt–response context. The actor microbatch uses the same token budget.
See the [training settings](README.md#training-settings) for sampling scales
and the actor learning rate.
