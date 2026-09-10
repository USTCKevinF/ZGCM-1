"""Runnable dataset-specific Pretrain profiles."""

from __future__ import annotations

from dataclasses import dataclass

from ....core import Step
from .finepdfs import FINEPDFS_STEPS
from .gharchive import GHARCHIVE_STEPS
from .olmocr import OLMOCR_STEPS


@dataclass(frozen=True, slots=True)
class PretrainDatasetProfile:
    source: str
    steps: tuple[Step, ...]


DATASET_PROFILES = {
    "finepdfs": PretrainDatasetProfile("pdf_ocr", FINEPDFS_STEPS),
    "gharchive": PretrainDatasetProfile("code", GHARCHIVE_STEPS),
    "olmocr": PretrainDatasetProfile("pdf_ocr", OLMOCR_STEPS),
}


def dataset_steps(profile: str | None, source: str) -> tuple[Step, ...]:
    if not profile:
        return ()
    key = profile.strip().casefold()
    if key not in DATASET_PROFILES:
        raise ValueError(f"unknown Pretrain dataset profile: {profile!r}")
    spec = DATASET_PROFILES[key]
    if source != spec.source:
        raise ValueError(f"Pretrain profile {key!r} requires source {spec.source!r}, not {source!r}")
    return spec.steps


__all__ = ["DATASET_PROFILES", "PretrainDatasetProfile", "dataset_steps"]
