# vision/selection.py
"""
Выбор целевого человека и расчёт коррекций наведения.
Чистые функции, ничего не знают про робота.
"""
import math
from typing import List, Optional

from .types import Detection


def score_detection(det: Detection, frame_w: int, frame_h: int) -> float:
    """Приоритет цели: больше — важнее.

    Без стерео/лидара абсолютной дистанции нет, но в indoor-сцене
    площадь bbox — хороший прокси близости. Дополнительно поощряем
    центрированность (легче наводиться) и низкое положение ступней
    (в комнате с видимым полом ближний человек стоит ниже в кадре).
    """
    area_norm = det.area / float(frame_w * frame_h)

    dx = abs(det.center_x - frame_w / 2) / (frame_w / 2)
    dy = abs(det.center_y - frame_h / 2) / (frame_h / 2)
    centering = 1.0 - min(1.0, math.hypot(dx, dy) / 1.414)

    feet_low = det.y_max / float(frame_h)

    return 0.60 * area_norm + 0.25 * centering + 0.15 * feet_low


def pick_closest_person(dets: List[Detection],
                        frame_w: int, frame_h: int) -> Optional[Detection]:
    """Самая приоритетная детекция или None."""
    if not dets:
        return None
    return max(dets, key=lambda d: score_detection(d, frame_w, frame_h))


def horizontal_correction(det: Detection, frame_w: int,
                          dead_zone_px: int) -> int:
    """+1 — крутить вправо, -1 — влево, 0 — цель в мёртвой зоне."""
    err = det.center_x - frame_w // 2
    if abs(err) <= dead_zone_px:
        return 0
    return +1 if err > 0 else -1


def vertical_correction(det: Detection, frame_h: int,
                        edge_margin_px: int) -> int:
    """+1 — наклонить вверх, -1 — вниз, 0 — не трогать.

    Логика: если bbox касается верхнего края, значит голова уходит
    за кадр — надо посмотреть выше (наклонить вверх, +1).
    Если касается нижнего — человек уходит за низ кадра,
    надо посмотреть ниже (наклонить вниз, -1).
    """
    if det.y_min <= edge_margin_px:
        return +1
    if det.y_max >= frame_h - edge_margin_px:
        return -1
    return 0