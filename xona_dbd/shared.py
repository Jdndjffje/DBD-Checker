from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class Circle:
    x: float
    y: float
    radius: float
    score: float = 0.0
    source: str = "unknown"


@dataclass
class Geometry:
    needle_angle: Optional[float] = None
    good_angle: Optional[float] = None
    great_angle: Optional[float] = None
    press_angle: Optional[float] = None
    target_type: str = "NONE"
    press_ready: bool = False
    needle_confidence: float = 0.0
    zone_confidence: float = 0.0
    good_width: float = 0.0
    great_width: float = 0.0
    simulated_press: bool = False


@dataclass
class RuntimeSnapshot:
    state: str = "IDLE"
    status: str = "Waiting"
    details: str = "—"
    circle: Optional[Circle] = None
    geometry: Optional[Geometry] = None
    candidate_count: int = 0
    best_score: float = 0.0
    debug_candidates: object = None
