# domain/constants.py
"""
Единый источник калибровочных значений.
Никаких зависимостей — чистые данные.
"""

# Направления (dy, dx) — экранная система, y вниз
DIRECTIONS = {
    "NORTH": (-1, 0),
    "EAST":  (0, 1),
    "SOUTH": (1, 0),
    "WEST":  (0, -1),
}
DIR_ORDER = ["NORTH", "EAST", "SOUTH", "WEST"]

# Время манёвров, секунды
SEC_PER_CELL     = 0.6
SEC_PER_90_TURN  = 0.4
SEC_PER_90_ARC_TURN = 1.9
SEC_STOP_PAUSE   = 0.5
SEC_ARRIVAL_STOP = 0.5
SEC_CAMERA_SWEEP = 0.5
AUTO_RETURN_WAIT_SEC = 3.0

DEFAULT_FREE = 10
PRIORITY     = 1
OBSTACLE     = 999