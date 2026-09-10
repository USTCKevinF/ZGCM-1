# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

import atexit, json
from collections import Counter
from typing import Any, Dict, Optional

import numpy as np
import torch

from megatron.core.datasets.gpt_dataset import GPTDatasetConfig
from megatron.core.datasets.megatron_dataset import LowLevelDataset, MegatronDataset
from megatron.core.datasets.utils import Split

IGNORE_INDEX = -100


class SFTLowLevelDataset:
    """The low-level dataset loading jsonl data for SFT

    Args:
        dataset_path (str): The path to jsonl data
            Each line of the jsonl must have key "messages" (List[Dict]),
            which is a sequence of system/user/assistant messages.
            Must be in the following format:
            [
                {"role": "system", "content": "something"},
                {"role": "user", "content": "something1"},
                {"role": "assistant", "content": "something2"},
            ]
            A jsonl line can contain multiple conversations packed together into on list. Each
            conversation starts with the system role, and conversations can have multiple turns
            of the user and assistant roles.
    """

    def __init__(self, dataset_path: str, cache_dir: str | None = None) -> None:
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "SFTDataset currently requires datasets library to be installed"
            )
        self.dataset = load_dataset("json", data_files=dataset_path, split="all", cache_dir=cache_dir)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> list:
        return self.dataset[idx]["messages"]


class PrepackedSFTLowLevelDataset:
    """Low-level dataset for pre-tokenized and pre-packed SFT JSONL.

    Each line must contain:
      format='megatron_sft_prepacked_v1', tokens, targets, cu_seqlens.
    Chat template and special-token handling are already applied upstream.
    """

    def __init__(self, dataset_path: str, cache_dir: str | None = None) -> None:
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "Prepacked SFTDataset currently requires datasets library to be installed"
            )
        self.dataset = load_dataset("json", data_files=dataset_path, split="all", cache_dir=cache_dir)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        return self.dataset[idx]


class IndexedPrepackedSFTLowLevelDataset:
    """Low-level dataset for Megatron indexed prepacked SFT streams.

    The dataset_path is a prefix. The following aligned indexed datasets must exist:
      {prefix}_tokens.{bin,idx}
      {prefix}_targets.{bin,idx}
      {prefix}_cu_seqlens.{bin,idx}
    """

    def __init__(self, dataset_path: str) -> None:
        from megatron.core.datasets.indexed_dataset import IndexedDataset

        self.prefix = dataset_path
        self.tokens = IndexedDataset(dataset_path + "_tokens")
        self.targets = IndexedDataset(dataset_path + "_targets")
        self.cu_seqlens = IndexedDataset(dataset_path + "_cu_seqlens")
        if len(self.tokens) != len(self.targets) or len(self.tokens) != len(self.cu_seqlens):
            raise ValueError(
                f"indexed prepacked stream length mismatch for {dataset_path}: "
                f"tokens={len(self.tokens)} targets={len(self.targets)} cu={len(self.cu_seqlens)}"
            )

    def __len__(self) -> int:
        return len(self.tokens)

    def __getitem__(self, idx: int) -> dict:
        return {
            "format": "megatron_sft_prepacked_v1",
            "tokens": self.tokens[idx],
            "targets": self.targets[idx],
            "cu_seqlens": self.cu_seqlens[idx],
        }


def _is_indexed_prepacked_sft_prefix(dataset_path: str) -> bool:
    from megatron.core.datasets.indexed_dataset import IndexedDataset

    return (
        IndexedDataset.exists(dataset_path + "_tokens")
        and IndexedDataset.exists(dataset_path + "_targets")
        and IndexedDataset.exists(dataset_path + "_cu_seqlens")
    )


def _is_prepacked_sft_file(dataset_path: str) -> bool:
    path = dataset_path.split()[0] if isinstance(dataset_path, str) else dataset_path
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                return False
            return (
                obj.get("format") == "megatron_sft_prepacked_v1"
                and "tokens" in obj
                and "targets" in obj
                and "cu_seqlens" in obj
            )
    return False


class SFTDataset(MegatronDataset):
    """The dataset used during SFT"""

    def __init__(
        self,
        dataset: LowLevelDataset,
        dataset_path: Optional[str],
        indices: np.ndarray,
        num_samples: Optional[int],
        index_split: Split,
        config: GPTDatasetConfig,
    ) -> None:
        super().__init__(dataset, dataset_path, indices, num_samples, index_split, config)

    @staticmethod
    def numel_low_level_dataset(low_level_dataset: LowLevelDataset) -> int:
        return len(low_level_dataset)

    @staticmethod
    def build_low_level_dataset(dataset_path: str, config: GPTDatasetConfig) -> LowLevelDataset:
        if _is_indexed_prepacked_sft_prefix(dataset_path):
            return IndexedPrepackedSFTLowLevelDataset(dataset_path)
        if _is_prepacked_sft_file(dataset_path):
            return PrepackedSFTLowLevelDataset(dataset_path, cache_dir=config.path_to_cache)
        return SFTLowLevelDataset(dataset_path)

    def __len__(self) -> int:
        if self.num_samples is None:
            return len(self.indices)
        return self.num_samples

    def _split_conversations(self, merged_conversations):
        split_conversations = []
        current = []
        for msg in merged_conversations:
            # Whenever we see a new system message, start a new conversation
            if msg["role"] == "system":
                if current:  # If previously accumulating a conversation, then store it
                    split_conversations.append(current)
                current = [msg]  # Then start the new conversation
            else:
                current.append(msg) # Continue accumulating the current conversation
        if current:  # Store any remaining conversation
            split_conversations.append(current)
        return split_conversations

    def _getitem_prepacked(self, record: Dict[str, Any], pack_length: int, pad: int) -> Dict[str, Any]:
        pack_tokens = [int(x) for x in record["tokens"]]
        pack_targets = [int(x) for x in record["targets"]]
        cu_seqlens_list = [int(x) for x in record["cu_seqlens"]]
        original_pack_length = len(pack_tokens)

        if len(pack_tokens) != len(pack_targets):
            raise ValueError("prepacked tokens and targets must have the same length")
        if len(pack_tokens) > pack_length + 1:
            raise ValueError(
                f"prepacked sample length {len(pack_tokens)} exceeds seq_length+1 {pack_length + 1}; "
                "refusing to truncate"
            )
        if len(cu_seqlens_list) < 2 or cu_seqlens_list[0] != 0:
            raise ValueError(f"bad prepacked cu_seqlens: {cu_seqlens_list[:8]}")
        if cu_seqlens_list[-1] != len(pack_tokens):
            raise ValueError(
                f"prepacked cu_seqlens[-1]={cu_seqlens_list[-1]} does not match token length {len(pack_tokens)}"
            )

        pack_positions = [0] * len(pack_tokens)
        for start, end in zip(cu_seqlens_list[:-1], cu_seqlens_list[1:]):
            if end < start:
                raise ValueError(f"non-monotonic prepacked cu_seqlens: {cu_seqlens_list[:16]}")
            pack_positions[start:end] = range(end - start)

        if len(pack_tokens) < pack_length + 1:
            pad_len = pack_length + 1 - len(pack_tokens)
            next_pos = (pack_positions[-1] + 1) if pack_positions else 0
            pack_tokens.extend([pad] * pad_len)
            pack_targets.extend([pad] * pad_len)
            pack_positions.extend(range(next_pos, next_pos + pad_len))

        assert len(pack_tokens) == pack_length + 1
        assert len(pack_targets) == pack_length + 1
        assert len(pack_positions) == pack_length + 1

        # Match the existing SFTDataset THD convention: final sequence covers
        # any remaining right-padding. This keeps TE shape assumptions stable.
        cu_seqlens_list[-1] = pack_length

        input_ids    = torch.tensor(pack_tokens[:-1],  dtype=torch.int64)
        labels       = torch.tensor(pack_targets[1:], dtype=torch.int64)
        position_ids = torch.tensor(pack_positions[:-1], dtype=torch.int64)

        loss_mask = torch.ones(pack_length, dtype=torch.float32)
        if original_pack_length <= pack_length:
            loss_mask[original_pack_length - 1 :] = 0.0
        loss_mask[labels == IGNORE_INDEX] = 0.0

        assert not self.config.create_attention_mask and not self.config.reset_attention_mask

        cu_seqlens = torch.tensor(cu_seqlens_list, dtype=torch.int32)
        adjacent_diffs = cu_seqlens[1:] - cu_seqlens[:-1]
        max_seqlen = adjacent_diffs.max()
        cu_seqlens_padded = cu_seqlens

        return {
            'tokens': input_ids,
            'labels': labels,
            'loss_mask': loss_mask,
            'position_ids': position_ids,
            'cu_seqlens': cu_seqlens,
            'cu_seqlens_padded': cu_seqlens_padded,
            'max_seqlen': max_seqlen,
        }

    def __getitem__(self, idx: int) -> Dict[str, Any]:

        tokenizer = self.config.tokenizer
        pack_length = self.config.sequence_length

        merged_conversations = self.dataset[int(self.indices[idx % len(self.indices)])]
        if isinstance(merged_conversations, dict) and merged_conversations.get("format") == "megatron_sft_prepacked_v1":
            pad = getattr(tokenizer, "pad", getattr(tokenizer, "pad_id"))
            return self._getitem_prepacked(merged_conversations, pack_length, pad)

        split_conversations = self._split_conversations(merged_conversations)

        def extend_with_padding(tokens, targets, positions, pad_len):
            tokens.extend([pad] * pad_len)
            targets.extend([pad] * pad_len)
            positions.extend(range(positions[-1]+1, positions[-1]+1+pad_len))

        pack_tokens = []
        pack_targets = []
        pack_positions = []
        cu_seqlens = [0]
        eod = tokenizer.eod
        pad = tokenizer.pad
        # TODO(duncan): Track number of convs dropped and/or truncated and amount of end-padding
        for conversation in split_conversations:

            tokens, targets = tokenizer.tokenize_conversation(
                conversation, return_target=True, add_generation_prompt=False
            )

            tokens_list = tokens.tolist()
            targets_list = targets.tolist()


            pack_tokens.extend(tokens_list)
            pack_targets.extend(targets_list)

            assert not self.config.reset_position_ids
            pack_positions.extend(range(len(tokens_list)))

            if self.config.context_parallel_size > 1:
                pad_granularity = self.config.context_parallel_size * 2
                mod_token_count = len(pack_tokens) % pad_granularity
                if mod_token_count != 0:
                    pad_len = pad_granularity - mod_token_count
                    extend_with_padding(pack_tokens, pack_targets, pack_positions, pad_len)

            # TODO(duncan): Consider also padding to multiple of number of tokens here. This might
            # be needed for efficiency (and potentially set via command-line argument).

            cu_seqlens.append(len(pack_tokens))

            # Handle any necessary truncation
            if len(pack_tokens) >= pack_length + 1:  # +1 here to account for later alignment
                # Truncate on the right
                max_body = pack_length
                pack_tokens = pack_tokens[:max_body]
                pack_targets = pack_targets[:max_body]
                pack_tokens.append(pad)
                pack_targets.append(pad)
                pack_positions = pack_positions[:pack_length+1]
                # Note len({pack_tokens, pack_targets, pack_positions}) should be pack_length + 1
                cu_seqlens[-1] = len(pack_tokens) - 1
                break

        # Handle any necessary padding
        if len(pack_tokens) < pack_length + 1:  # +1 here to account for later alignment
            pad_len = pack_length + 1 - len(pack_tokens)
            extend_with_padding(pack_tokens, pack_targets, pack_positions, pad_len)
            # Note len({pack_tokens, pack_targets, pack_positions}) should be pack_length + 1
            cu_seqlens[-1] = len(pack_tokens) - 1

        assert len(pack_tokens) == pack_length + 1
        assert len(pack_targets) == pack_length + 1
        assert len(pack_positions) == pack_length + 1

        # Align and convert to tensors
        input_ids    = torch.tensor(pack_tokens[:-1],  dtype=torch.int64)
        labels       = torch.tensor(pack_targets[1:], dtype=torch.int64)
        position_ids = torch.tensor(pack_positions[:-1], dtype=torch.int64)

        # Loss mask.
        loss_mask = torch.ones(pack_length, dtype=torch.float32)
        loss_mask[labels == pad] = 0.0  # Mask paddings
        loss_mask[labels == IGNORE_INDEX] = 0.0  # mask prompts

        # TODO(duncan): Optionally create an attention mask
        assert not self.config.create_attention_mask and not self.config.reset_attention_mask
        # attention_mask = None

        assert len(cu_seqlens) >= 2
        cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32)
        # Calculating max_seqlen here, rather than incrementally above, because of possible
        # effects of truncation and padding
        adjacent_diffs = cu_seqlens[1:] - cu_seqlens[:-1]
        max_seqlen = adjacent_diffs.max()  # max_seqlen is a 0-D tensor
        # THD context-parallel partitioning expects padded sequence offsets.
        # The SFT packer already pads conversation boundaries for CP above, so
        # the packed offsets are also the padded offsets for this format.
        cu_seqlens_padded = cu_seqlens

        return {
            'tokens': input_ids,
            'labels': labels,
            # 'attention_mask': attention_mask,  # PyTorch collate cannot handle NoneType
            'loss_mask': loss_mask,
            'position_ids': position_ids,
            'cu_seqlens': cu_seqlens,
            'cu_seqlens_padded': cu_seqlens_padded,
            'max_seqlen': max_seqlen,
        }
