"""Shared exact and optional near-deduplication operations."""

from ..dedup import NearDeduper
from ..steps.common import exact_deduplicate, near_deduplicate

__all__ = ["NearDeduper", "exact_deduplicate", "near_deduplicate"]
