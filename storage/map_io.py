# storage/map_io.py
"""
Сохранение и загрузка карты в формате maps/<name>.json.
Не знает про UI, поиск пути и класс PrecomputedGymMap.
"""
import os
import json
import numpy as np

from domain.grid import GridMap
from .json_utils import to_jsonable


def save_map(grid, robot_pos, robot_home, robot_heading, maps_dir, name="map_1"):
    """Сохраняет карту в maps/<name>.json. Возвращает путь к файлу."""
    os.makedirs(maps_dir, exist_ok=True)
    filepath = os.path.join(maps_dir, f"{name}.json")

    data = {
        "width": int(grid.width),
        "height": int(grid.height),
        "matrix": grid.matrix.tolist(),
        "obstacles_registry": {
            str(k): [list(c) for c in v]
            for k, v in grid.obstacles.items()
        },
        "slots_registry": {
            str(k): list(v) for k, v in grid.slots.items()
        },
        "robot_pos": [int(robot_pos[0]), int(robot_pos[1])],
        "robot_home": [int(robot_home[0]), int(robot_home[1])],
        "robot_heading": robot_heading,
        "constants": {
            "DEFAULT_FREE": int(grid.DEFAULT_FREE),
            "PRIORITY": int(grid.PRIORITY),
            "OBSTACLE": int(grid.OBSTACLE),
        },
    }
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(data), f, indent=4, ensure_ascii=False)
    print(f"[СИСТЕМА] Карта сохранена: {filepath}")
    return filepath


def load_map(maps_dir, name="map_1"):
    """
    Загружает карту из maps/<name>.json.
    Возвращает (grid, robot_pos, robot_home, robot_heading, raw_data).
    raw_data — исходный словарь из файла, если кому-то понадобится.
    """
    filepath = os.path.join(maps_dir, f"{name}.json")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Карта не найдена: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    grid = GridMap(
        width=data["width"],
        height=data["height"],
        matrix=np.array(data["matrix"], dtype=int),
        obstacles={int(k): [tuple(c) for c in v]
                   for k, v in data["obstacles_registry"].items()},
        slots={int(k): tuple(v)
               for k, v in data["slots_registry"].items()},
    )
    for obs_id, cells in grid.obstacles.items():
        for cell in cells:
            grid.cell_to_obs[cell] = obs_id

    robot_pos     = tuple(data["robot_pos"])
    robot_home    = tuple(data["robot_home"])
    robot_heading = data["robot_heading"]

    print(f"[СИСТЕМА] Карта загружена: {filepath}")
    return grid, robot_pos, robot_home, robot_heading, data