# domain/commands.py
"""
Преобразование пути в команды для робота.
Чистые функции: никаких зависимостей от UI, файлов, класса карты.
"""
from .constants import (
    DIRECTIONS, DIR_ORDER,
    SEC_PER_CELL, SEC_PER_90_TURN, SEC_PER_90_ARC_TURN, SEC_STOP_PAUSE,
    SEC_ARRIVAL_STOP, SEC_CAMERA_SWEEP,
)


def angle_from_heading(heading: str) -> float:
    """0=N, 90=E, 180=S, 270=W (экранные градусы, y вниз)."""
    return float(DIR_ORDER.index(heading) * 90)


def _delta_to_heading(delta):
    for name, d in DIRECTIONS.items():
        if d == delta:
            return name
    return None


def arrival_search_sequence():
    """
    Хвостовая последовательность «поиска тренажёра»:
    stealth on -> камера вправо -> центр -> камера влево -> центр.
    """
    return [
        {"action": "stealth",  "on": True, "duration": 0.0},
        {"action": "camera-h", "dir":  1,  "duration": SEC_CAMERA_SWEEP},
        {"action": "camera-h", "dir":  0},
        {"action": "camera-h", "dir": -1,  "duration": SEC_CAMERA_SWEEP},
        {"action": "camera-h", "dir":  0},
    ]


def path_to_commands(path, initial_heading="NORTH"):
    """
    Последовательность координат -> (commands, final_heading).
    final_heading нужен, чтобы после миссии обновить robot_heading.

    Каждая команда:
      {"action": "drive", "wheel": ..., "steer": ..., "slow": ..., "duration": ...}
      {"action": "stop",  "duration": ...}
      {"action": "stealth", "on": True, "duration": 0.0}
      {"action": "camera-h", "dir": ..., "duration": ...}
    """
    def drive_forward(duration):
        return {"action": "drive", "wheel": 1, "steer": 0,
                "slow": False, "duration": round(duration, 3)}

    def turn_arc(steer_sign, duration):
        """Поворот с движением: колёса вывернуты + робот едет.
        На Ackermann-шасси поворот на месте невозможен."""
        return {"action": "drive", "wheel": 1, "steer": steer_sign,
                "slow": True, "duration": round(duration, 3)}

    def stop(duration=SEC_STOP_PAUSE):
        return {"action": "stop", "duration": round(duration, 3)}

    commands = []
    if not path or len(path) < 2:
        # Пустой путь — просто поиск на месте
        commands.extend(arrival_search_sequence())
        return commands, initial_heading

    heading = initial_heading
    for i in range(len(path) - 1):
        dy = path[i + 1][0] - path[i][0]
        dx = path[i + 1][1] - path[i][1]
        new_heading = _delta_to_heading((dy, dx))
        if new_heading is None:
            continue  # диагоналей в 4-связном A* быть не должно

        if new_heading != heading:
            diff = (DIR_ORDER.index(new_heading)
                    - DIR_ORDER.index(heading)) % 4

            commands.append(stop(SEC_STOP_PAUSE))

            if diff == 1:                     # вправо на 90°
                commands.append(turn_arc(+1, SEC_PER_90_ARC_TURN))
            elif diff == 3:                   # влево на 90°
                commands.append(turn_arc(-1, SEC_PER_90_ARC_TURN))
            elif diff == 2:                   # разворот на 180°
                # Одна длинная дуга вправо — проще и предсказуемее,
                # чем две короткие
                commands.append(turn_arc(+1, 2.0 * SEC_PER_90_ARC_TURN))

            heading = new_heading

            # Если предыдущая команда была arc-turn, часть клетки уже проехали.
            # Оставшуюся часть компенсируем укороченным drive.
            if commands and commands[-1].get("action") == "drive" \
               and commands[-1].get("steer", 0) != 0:
                commands.append(drive_forward(SEC_PER_CELL * 0.7))
            else:
                commands.append(drive_forward(SEC_PER_CELL))

    commands.append(stop(SEC_ARRIVAL_STOP))
    commands.extend(arrival_search_sequence())
    return commands, heading

# ---------- Расписание для анимации ----------

def build_animation_schedule(commands, start_pos, start_heading):
    """
    Разворачивает список команд в таймлайн движений.
    Возвращает (segments, total_time).
    segment = {"t0": float, "t1": float,
               "state0": (x, y, angle_deg),
               "state1": (x, y, angle_deg)}
    """
    x, y = float(start_pos[1]), float(start_pos[0])
    angle = angle_from_heading(start_heading)
    heading_idx = DIR_ORDER.index(start_heading)

    segments = []
    t = 0.0

    def push(dur, x0, y0, a0, x1, y1, a1):
        nonlocal t
        segments.append({
            "t0": t, "t1": t + dur,
            "state0": (x0, y0, a0),
            "state1": (x1, y1, a1),
        })
        t += dur

    for cmd in commands:
        action = cmd.get("action")
        dur = float(cmd.get("duration", 0.0) or 0.0)

        if action == "drive" and cmd.get("wheel") == 1 and cmd.get("steer", 0) == 0:
            # Едем вперёд на одну клетку
            dy, dx = DIRECTIONS[DIR_ORDER[heading_idx]]
            nx, ny = x + dx, y + dy
            push(dur, x, y, angle, nx, ny, angle)
            x, y = nx, ny

        elif action == "drive" and cmd.get("wheel", 0) == 0 and cmd.get("steer", 0) != 0:
            # Поворот на месте
            steer = cmd["steer"]
            new_angle = angle + (90.0 if steer > 0 else -90.0)
            push(dur, x, y, angle, x, y, new_angle)
            angle = new_angle
            heading_idx = (heading_idx + (1 if steer > 0 else -1)) % 4

        elif dur > 0:
            # stop / stealth / camera-h — просто «тикаем» время на месте
            push(dur, x, y, angle, x, y, angle)

    return segments, t


def state_at_time(schedule, t):
    """Возвращает (x, y, angle_deg) в момент времени t (с линейной интерполяцией)."""
    if not schedule:
        return (0.0, 0.0, 0.0)

    first, last = schedule[0], schedule[-1]
    if t <= first["t0"]:
        return first["state0"]
    if t >= last["t1"]:
        return last["state1"]

    for seg in schedule:
        if seg["t0"] <= t <= seg["t1"]:
            span = seg["t1"] - seg["t0"]
            if span <= 0:
                return seg["state1"]
            a = (t - seg["t0"]) / span
            s0, s1 = seg["state0"], seg["state1"]
            return (
                s0[0] + (s1[0] - s0[0]) * a,
                s0[1] + (s1[1] - s0[1]) * a,
                s0[2] + (s1[2] - s0[2]) * a,
            )
    return last["state1"]