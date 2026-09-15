"""AReaL-compatible math reward exports and dispatcher."""

from __future__ import annotations

import asyncio
from typing import Any

from rewards.math_reward import math_reward_fn


async def async_reward_router_fn(
    prompt: str,
    completions: str,
    prompt_ids: list[int] | None = None,
    completion_ids: list[int] | None = None,
    **kwargs: Any,
) -> float:
    """Non-blocking variant for custom async rollout workflows."""

    return await asyncio.to_thread(
        math_reward_fn,
        prompt,
        completions,
        prompt_ids,
        completion_ids,
        **kwargs,
    )


__all__ = [
    "async_reward_router_fn",
    "math_reward_fn",
]
