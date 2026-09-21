"""
Модель карты зала: матрица + реестры тренажёров.
Никаких зависимостей от UI, файлов, поиска пути.
"""
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from .constants import DEFAULT_FREE, PRIORITY, OBSTACLE


@dataclass
class GridMap:
    width: int
    height: int
    matrix: np.ndarray
    obstacles:   dict = field(default_factory=dict)   # {obs_id: [(y, x), ...]}
    slots:       dict = field(default_factory=dict)   # {obs_id: (y, x)}
    cell_to_obs: dict = field(default_factory=dict)   # {(y, x): obs_id}

    # Класс-константы — для pathfinder'а, чтобы он читал grid.OBSTACLE и т.п.
    DEFAULT_FREE = DEFAULT_FREE
    PRIORITY     = PRIORITY
    OBSTACLE     = OBSTACLE

    @classmethod
    def empty(cls, width: int, height: int) -> "GridMap":
        return cls(
            width=width,
            height=height,
            matrix=np.full((height, width), DEFAULT_FREE, dtype=int),
        )

    # ---------- запросы ----------
    def in_bounds(self, y: int, x: int) -> bool:
        return 0 <= y < self.height and 0 <= x < self.width

    def obs_at(self, y: int, x: int) -> Optional[int]:
        return self.cell_to_obs.get((y, x))

    # ---------- редактирование ----------
    def add_apparatus(self, obs_id, top_y, top_x, h, w, slot_y, slot_x):
        if top_y < 0 or top_x < 0 or top_y + h > self.height or top_x + w > self.width:
            raise ValueError(f"Тренажер #{obs_id}: габарит выходит за границы карты.")
        if not (0 <= slot_y < self.height and 0 <= slot_x < self.width):
            raise ValueError(f"Тренажер #{obs_id}: слот ({slot_y},{slot_x}) вне границ карты.")
        if top_y <= slot_y < top_y + h and top_x <= slot_x < top_x + w:
            raise ValueError(f"Тренажер #{obs_id}: слот внутри габарита тренажера.")
        if self.matrix[slot_y, slot_x] == OBSTACLE:
            raise ValueError(f"Тренажер #{obs_id}: слот ({slot_y},{slot_x}) занят другим препятствием.")

        if obs_id in self.obstacles:
            self.remove_apparatus(obs_id)

        cells = []
        for y in range(top_y, top_y + h):
            for x in range(top_x, top_x + w):
                self.matrix[y, x] = OBSTACLE
                self.cell_to_obs[(y, x)] = obs_id
                cells.append((y, x))

        self.obstacles[obs_id] = cells
        self.slots[obs_id] = (slot_y, slot_x)

    def remove_apparatus(self, obs_id):
        for (y, x) in self.obstacles.pop(obs_id, []):
            if self.matrix[y, x] == OBSTACLE:
                self.matrix[y, x] = DEFAULT_FREE
            self.cell_to_obs.pop((y, x), None)
        self.slots.pop(obs_id, None)

    def def_priority_way(self, start_y, start_x, end_y, end_x):
        y_min, y_max = min(start_y, end_y), max(start_y, end_y)
        x_min, x_max = min(start_x, end_x), max(start_x, end_x)
        if y_min < 0 or x_min < 0 or y_max >= self.height or x_max >= self.width:
            raise ValueError(f"Дорожка ({start_y},{start_x})->({end_y},{end_x}) "
                             f"выходит за границы карты.")
        for y in range(y_min, y_max + 1):
            for x in range(x_min, x_max + 1):
                if self.matrix[y, x] != OBSTACLE:
                    self.matrix[y, x] = PRIORITY