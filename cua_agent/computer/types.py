from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DisplayInfo:
    logical_width: int
    logical_height: int
    physical_width: int
    physical_height: int
    scale_factor: float

