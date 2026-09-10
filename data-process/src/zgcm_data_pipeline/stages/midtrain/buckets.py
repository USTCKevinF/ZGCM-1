"""Length-bucket primitives used by Midtrain mix construction."""

from __future__ import annotations


def assign_length_bucket(token_count: int, boundaries: tuple[int, ...] = (16_384, 65_536, 262_144)) -> str:
    """Return a stable base bucket label for a token length."""
    if not boundaries or any(int(boundary) <= 0 for boundary in boundaries):
        raise ValueError("boundaries must contain positive values")
    value = int(token_count)
    lower = 0
    for upper in boundaries:
        upper = int(upper)
        if value <= upper:
            return f"B{upper // 1024}"
        lower = upper
    return f"gt_{lower // 1024}"


__all__ = ["assign_length_bucket"]
