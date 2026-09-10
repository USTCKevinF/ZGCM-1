# Environment

The mixed-RL experiments in Section 4.2 of the technical report use a policy
model and tokenizer, a mixture of mathematics/code/general prompts, and
domain-specific reward evaluation.

Experiment inputs include:

- Policy weights and the associated tokenizer.
- Prompts selected using initial-policy difficulty estimates.
- Answer verifiers for mathematics and correctness-based general tasks.
- Executable tests for code rewards.
- Instruction-following criteria for the corresponding general tasks.
- A reference policy for experiments with KL regularization.

Rollouts allow up to 65,536 generated tokens within a 98,304-token total
prompt–response context. The actor microbatch uses the same token budget.
See the [training settings](README.md#training-settings) for sampling scales
and the actor learning rate.
