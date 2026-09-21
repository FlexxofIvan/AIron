# services/route_service.py
"""
Оркестрация маршрутов: пакетный предрасчёт от док-станции ко всем тренажёрам
и построение одиночного маршрута «домой».

Не знает про matplotlib и класс PrecomputedGymMap.
Принимает GridMap и пути к папкам.
"""
from domain.pathfinder import dijkstra_from, astar, reconstruct_path
from domain.commands import path_to_commands
from storage import routes_io


def build_all_routes(grid, robot_home, robot_heading, routes_dir, commands_dir):
    """
    Один Dijkstra от robot_home, нарезка маршрутов ко всем слотам.
    Для каждого тренажёра сохраняет:
      routes/route_<id>.json           — метаданные (включая обратный маршрут)
      commands/command_<id>.json       — команды «дом → слот»
      commands/command_<id>_back.json  — команды «слот → дом», с учётом курса прибытия
    Возвращает {obs_id: route_dict}.
    """
    print("[ПЛАНИРОВЩИК] Предрасчёт маршрутов...")
    came_from, dist = dijkstra_from(grid, robot_home)
    database = {}

    for obs_id, target_slot in grid.slots.items():
        path = reconstruct_path(came_from, robot_home, target_slot)
        if not path:
            print(f" -> [ОШИБКА] Тренажер #{obs_id} заблокирован! Путь не найден.")
            continue

        # Прямой маршрут: home → slot
        commands, final_heading = path_to_commands(path, robot_heading)

        # Обратный маршрут: тот же путь, но в обратную сторону.
        # Начальный курс — final_heading (то, куда робот «смотрит» после прибытия).
        back_path = list(reversed(path))
        back_commands, back_final_heading = path_to_commands(back_path, final_heading)

        route = {
            "obs_id": int(obs_id),
            "start": [int(robot_home[0]), int(robot_home[1])],
            "target_slot": [int(target_slot[0]), int(target_slot[1])],
            "total_cost": int(dist[target_slot]) if target_slot in dist else None,
            # прямой
            "path_coordinates":       [[int(p[0]), int(p[1])] for p in path],
            "movement_commands":      commands,
            "final_heading":          final_heading,
            # обратный
            "return_path_coordinates": [[int(p[0]), int(p[1])] for p in back_path],
            "return_commands":         back_commands,
            "return_initial_heading":  final_heading,
            "return_final_heading":    back_final_heading,
        }
        database[obs_id] = route

        route_path   = routes_io.save_route(route, routes_dir)
        cmd_path     = routes_io.save_commands(commands,      obs_id, commands_dir)
        cmd_back     = routes_io.save_commands(back_commands, obs_id, commands_dir,
                                               suffix="_back")
        print(f" -> Маршрут #{obs_id}:")
        print(f"      прямой:  {cmd_path}")
        print(f"      обратно: {cmd_back}")

    print(f"[СИСТЕМА] Сохранено маршрутов: {len(database)}.")
    return database


def build_return_home_route(grid, robot_home, from_pos, from_heading, commands_dir):
    """
    Одиночный A* от текущей позиции робота до док-станции.
    Сохраняет commands/command_home.json.

    Возвращает (path, commands, final_heading, command_file_path)
    или None, если робот уже дома или путь не найден.
    """
    if tuple(from_pos) == tuple(robot_home):
        print("[СИСТЕМА] Робот уже на док-станции.")
        return None

    path = astar(grid, tuple(from_pos), tuple(robot_home))
    if not path:
        print("[ОШИБКА] Путь домой не найден (заблокирован?).")
        return None

    commands, final_heading = path_to_commands(path, from_heading)
    cmd_file = routes_io.save_commands(commands, "home", commands_dir)
    print(f"[СИСТЕМА] Маршрут домой: {cmd_file}")
    return path, commands, final_heading, cmd_file

def build_route_to_target(grid, from_pos, from_heading, target_slot, commands_dir):
    """
    Маршрут от текущей позиции робота к целевому слоту.
    Сохраняет commands/command_current.json.
    Возвращает (path, commands, final_heading, command_file) или None.
    """
    from_pos    = tuple(from_pos)
    target_slot = tuple(target_slot)

    if from_pos == target_slot:
        path = [from_pos]
    else:
        path = astar(grid, from_pos, target_slot)
        if not path:
            print(f"[СИСТЕМА] Путь из {from_pos} в {target_slot} не найден.")
            return None

    commands, final_heading = path_to_commands(path, from_heading)
    cmd_file = routes_io.save_commands(commands, "current", commands_dir)
    return path, commands, final_heading, cmd_file