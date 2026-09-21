# main.py
"""
Точка входа приложения Rover Revolution.

Собирает демо-карту, считает маршруты, открывает интерактивное окно.
Позже здесь же появится запуск веб-сервера.
"""
import os

from domain.grid import GridMap
from storage import map_io
from services import route_service
from map import InteractiveMapUI


BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
MAPS_DIR     = os.path.join(BASE_DIR, "maps")
COMMANDS_DIR = os.path.join(BASE_DIR, "commands")
ROUTES_DIR   = os.path.join(BASE_DIR, "routes")


if __name__ == "__main__":
    # Демо-карта зала: два тренажёра и две приоритетные дорожки
    grid = GridMap.empty(40, 40)
    grid.add_apparatus(obs_id=1, top_y=3,  top_x=2,  h=3, w=3,
                       slot_y=6,  slot_x=3)
    grid.add_apparatus(obs_id=2, top_y=15, top_x=15, h=3, w=3,
                       slot_y=14, slot_x=16)
    grid.def_priority_way(start_y=6,  start_x=1, end_y=6,  end_x=19)
    grid.def_priority_way(start_y=14, start_x=1, end_y=14, end_x=19)

    robot_home    = (1, 1)
    robot_heading = "NORTH"

    # Сохраняем карту
    map_io.save_map(grid, robot_home, robot_home, robot_heading,
                    MAPS_DIR, "map_1")

    # Считаем и сохраняем маршруты ко всем тренажёрам
    routes_db = route_service.build_all_routes(
        grid, robot_home, robot_heading, ROUTES_DIR, COMMANDS_DIR
    )

    # Поднимаем UI
    ui = InteractiveMapUI(
        grid=grid,
        commands_dir=COMMANDS_DIR,
        robot_pos=robot_home,
        robot_home=robot_home,
        robot_heading=robot_heading,
        routes_database=routes_db,
    )
    ui.show_interactive_map()