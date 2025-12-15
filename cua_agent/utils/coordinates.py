"""Coordinate conversions between logical points and physical pixels."""

from __future__ import annotations

from typing import Tuple


def px_to_point(x: float, y: float, scale: float) -> Tuple[float, float]:
    """Convert physical pixels to logical points."""
    if scale <= 0:
        raise ValueError("scale must be > 0")
    return x / scale, y / scale


def point_to_px(x: float, y: float, scale: float) -> Tuple[int, int]:
    """Convert logical points to physical pixels."""
    if scale <= 0:
        raise ValueError("scale must be > 0")
    return int(round(x * scale)), int(round(y * scale))


def clamp_point(x: float, y: float, width: int, height: int) -> Tuple[float, float]:
    """Clamp logical coordinates to a rectangle."""
    clamped_x = max(0.0, min(x, float(width - 1)))
    clamped_y = max(0.0, min(y, float(height - 1)))
    return clamped_x, clamped_y

