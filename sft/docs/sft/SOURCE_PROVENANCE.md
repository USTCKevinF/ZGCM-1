# Source provenance

The training stack is based on NVIDIA Megatron-LM commit
`eba2eaf71d34274c1ca0a59ac5e57a6bf53eb732`.

The ZGCM-1 SFT implementation in this branch adds the GLM5.1 SFT dataset and
tokenizer contract, packed-sequence training inputs, variable-length FA3,
attention output gating, TP-aware Muon behavior, distributed checkpoint
compatibility, and the stable SFT launcher.
