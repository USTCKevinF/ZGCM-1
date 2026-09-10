"""Reusable, stage-independent data processing operations."""

from .dedup import NearDeduper, exact_deduplicate, near_deduplicate
from .io import iter_jsonl, open_text, write_jsonl
from .length import estimate_tokens_and_length_bucket
from .manifest import build_manifest, file_sha256, write_manifest
from .quality import QualityPolicy, apply_quality_policy
from .schema import normalize_schema_text, normalize_text_value, validate_basic_integrity, validate_provenance
from .shuffle import deterministic_shuffle_key, stable_partition
from .tokenize import TokenizedSample, build_loss_mask, pack_tokenized_samples, validate_tokenized_sample
from .work import RowMicroshard, WorkUnit, balance_work_units, plan_row_microshards
from .validate import summarize_decisions, validate_release_record

__all__ = [
    "NearDeduper",
    "QualityPolicy",
    "RowMicroshard",
    "TokenizedSample",
    "WorkUnit",
    "apply_quality_policy",
    "balance_work_units",
    "build_loss_mask",
    "build_manifest",
    "deterministic_shuffle_key",
    "estimate_tokens_and_length_bucket",
    "exact_deduplicate",
    "file_sha256",
    "iter_jsonl",
    "near_deduplicate",
    "normalize_schema_text",
    "normalize_text_value",
    "open_text",
    "pack_tokenized_samples",
    "plan_row_microshards",
    "stable_partition",
    "summarize_decisions",
    "validate_basic_integrity",
    "validate_provenance",
    "validate_release_record",
    "validate_tokenized_sample",
    "write_jsonl",
    "write_manifest",
]
