"""Runnable dataset-specific Midtrain profiles."""

from __future__ import annotations

from dataclasses import dataclass

from ....core import Step
from .agent_coding import AGENT_CODING_STEPS
from .nemotron_math import NEMOTRON_MATH_STEPS
from .web_knowledge import WEB_KNOWLEDGE_STEPS


@dataclass(frozen=True, slots=True)
class MidtrainDatasetProfile:
    category: str
    steps: tuple[Step, ...]


DATASET_PROFILES = {
    "agent_coding": MidtrainDatasetProfile("agentic", AGENT_CODING_STEPS),
    "nemotron_math": MidtrainDatasetProfile("math", NEMOTRON_MATH_STEPS),
    "web_knowledge": MidtrainDatasetProfile("web", WEB_KNOWLEDGE_STEPS),
}


def dataset_steps(profile: str | None, category: str) -> tuple[Step, ...]:
    if not profile:
        return ()
    key = profile.strip().casefold()
    if key not in DATASET_PROFILES:
        raise ValueError(f"unknown Midtrain dataset profile: {profile!r}")
    spec = DATASET_PROFILES[key]
    if category != spec.category:
        raise ValueError(f"Midtrain profile {key!r} requires category {spec.category!r}, not {category!r}")
    return spec.steps


__all__ = ["DATASET_PROFILES", "MidtrainDatasetProfile", "dataset_steps"]
