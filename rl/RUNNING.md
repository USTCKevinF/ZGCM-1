# Experiment Protocol

Section 4.2 of the ZGCM-1 technical report describes the following mixed-RL
procedure:

1. Estimate prompt difficulty using rollouts from the initial policy and remove
   problems that the model already solves frequently.
2. Sample response groups at either 24 prompts × 16 responses or
   384 prompts × 8 responses per training step.
3. Score responses with domain-specific rewards: answer correctness for
   mathematics, executable-test pass fraction for code, and correctness or
   instruction following for general tasks.
4. Filter and replace groups with zero reward variance through dynamic sampling.
5. Construct group-relative advantages and optimize the policy with GRPO at an
   actor learning rate of `2e-6`.

The experiments also explore reference-policy KL regularization and a mild
length penalty. Invalid or truncated responses receive no positive reward.
The response and total-context budgets are 65,536 and 98,304 tokens,
respectively.
