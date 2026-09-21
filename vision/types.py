# vision/types.py
"""
Иммутабельные/простые типы видео-пайплайна.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Chunk:
    """Кусок H.264-байт с робота (Annex B) + timestamp в мс от робота."""
    data: bytes
    ts_ms: int

    @property
    def ts_sec(self) -> float:
        return self.ts_ms / 1000.0


@dataclass
class Frame:
    """Декодированный кадр (BGR numpy) + timestamp в мс."""
    image: np.ndarray       # H×W×3, uint8, BGR
    ts_ms: int

    @property
    def ts_sec(self) -> float:
        return self.ts_ms / 1000.0

    @property
    def width(self) -> int:
        return int(self.image.shape[1])

    @property
    def height(self) -> int:
        return int(self.image.shape[0])


@dataclass(frozen=True)
class Detection:
    """Один детектированный объект в координатах кадра."""
    x_min: int
    y_min: int
    x_max: int
    y_max: int
    score: float

    @property
    def width(self) -> int:
        return self.x_max - self.x_min

    @property
    def height(self) -> int:
        return self.y_max - self.y_min

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def center_x(self) -> int:
        return (self.x_min + self.x_max) // 2

    @property
    def center_y(self) -> int:
        return (self.y_min + self.y_max) // 2