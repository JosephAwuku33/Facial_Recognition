from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass(frozen=True)
class FaceData:
    """Immutable Data Transfer Object for a detected face."""

    name: str
    location: Tuple[int, int, int, int]  # x, y, w, h
    distance: float # Euclidean distance from known face
