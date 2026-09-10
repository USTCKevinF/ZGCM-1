"""Runnable dataset-specific Posttrain profiles."""

from __future__ import annotations

from dataclasses import dataclass

from ....core import Step
from .dolci_think import DOLCI_THINK_STEPS
from .dolci_tooluse import DOLCI_TOOLUSE_STEPS
from .ultradata import ULTRADATA_STEPS


@dataclass(frozen=True, slots=True)
class PosttrainDatasetProfile:
    categories: tuple[str, ...]
    steps: tuple[Step, ...]


DATASET_PROFILES = {
    "dolci_think": PosttrainDatasetProfile(("instruction",), DOLCI_THINK_STEPS),
    "dolci_tooluse": PosttrainDatasetProfile(("agentic",), DOLCI_TOOLUSE_STEPS),
    "ultradata": PosttrainDatasetProfile(("instruction",), ULTRADATA_STEPS),
}


def dataset_steps(profile: str | None, category: str) -> tuple[Step, ...]:
    if not profile:
        return ()
    key = profile.strip().casefold()
    if key not in DATASET_PROFILES:
        raise ValueError(f"unknown Posttrain dataset profile: {profile!r}")
    spec = DATASET_PROFILES[key]
    if category not in spec.categories:
        raise ValueError(f"Posttrain profile {key!r} requires category in {spec.categories!r}, not {category!r}")
    return spec.steps


__all__ = ["DATASET_PROFILES", "PosttrainDatasetProfile", "dataset_steps"]
