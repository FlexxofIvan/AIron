# storage/routes_io.py
"""
Сохранение и загрузка маршрутов и команд робота.
Формат:
  routes/route_<id>.json     — метаданные маршрута
  commands/command_<id>.json — плоский список команд для main_rover.run
"""
import os
import json
from .json_utils import to_jsonable


def save_route(route, routes_dir):
    os.makedirs(routes_dir, exist_ok=True)
    path = os.path.join(routes_dir, f"route_{route['obs_id']}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(route), f, indent=4, ensure_ascii=False)
    return path


def save_commands(commands, obs_id, commands_dir, suffix=""):
    """suffix: добавочный суффикс в имени файла (например '_back' для обратного маршрута)."""
    os.makedirs(commands_dir, exist_ok=True)
    path = os.path.join(commands_dir, f"command_{obs_id}{suffix}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(commands), f, indent=4, ensure_ascii=False)
    return path


def load_all_routes(routes_dir):
    """Возвращает {obs_id: route_dict}."""
    if not os.path.isdir(routes_dir):
        print(f"[СИСТЕМА] Папка {routes_dir} не найдена, база путей пуста.")
        return {}

    db = {}
    for fname in sorted(os.listdir(routes_dir)):
        if not (fname.startswith("route_") and fname.endswith(".json")):
            continue
        filepath = os.path.join(routes_dir, fname)
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        obs_id = data.get("obs_id")
        if obs_id is None:
            try:
                obs_id = int(fname.replace("route_", "").replace(".json", ""))
            except ValueError:
                continue

        db[int(obs_id)] = data

    print(f"[СИСТЕМА] Загружено маршрутов: {len(db)} (папка {routes_dir}).")
    return db