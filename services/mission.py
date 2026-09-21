# services/mission.py
"""
Модель одной миссии: робот едет по маршруту.
Не знает про UI, файлы и matplotlib.
Знает про драйвер и расписание.
"""
import threading
import time
from enum import Enum

from domain.commands import build_animation_schedule, state_at_time


class MissionStatus(Enum):
    IDLE    = "idle"
    RUNNING = "running"
    DONE    = "done"
    FAILED  = "failed"


class Mission:
    def __init__(self, commands, start_pos, start_heading,
                 driver, target_pos, target_heading, command_file,
                 is_return_home=False):
        self.schedule, self.total_time = build_animation_schedule(
            commands, start_pos, start_heading
        )
        self.driver         = driver
        self.target_pos     = tuple(target_pos)
        self.target_heading = target_heading
        self.command_file   = command_file
        self.is_return_home = is_return_home      # ← новое

        self.started_at     = None
        self.status         = MissionStatus.IDLE
        self._driver_result = None
        self._thread        = None

    def start(self):
        """Запускает драйвер в фоновом потоке и включает таймлайн."""
        self.started_at = time.monotonic()
        self.status     = MissionStatus.RUNNING
        self._thread    = threading.Thread(target=self._run_driver, daemon=True)
        self._thread.start()

    def _run_driver(self):
        try:
            self.driver.execute(self.command_file)
            self._driver_result = "ok"
        except Exception as e:
            self._driver_result = f"error: {e}"

    def state_now(self):
        """Текущая позиция для отрисовки. None если миссия не RUNNING."""
        if self.status != MissionStatus.RUNNING or self.started_at is None:
            return None
        t = time.monotonic() - self.started_at
        return state_at_time(self.schedule, t)

    def elapsed(self) -> float:
        if self.started_at is None:
            return 0.0
        return time.monotonic() - self.started_at

    def is_finished(self) -> bool:
        """
        Миссия завершена, если:
          - драйвер вернул ошибку (анимация прерывается сразу), ИЛИ
          - драйвер успешно завершился И симуляция отыграла своё время.
        """
        if self._driver_result is None:
            return False
        if self._driver_result != "ok":
            return True
        return self.elapsed() >= self.total_time

    def finalize(self) -> MissionStatus:
        """Переводит миссию в DONE или FAILED. Вызывается UI-таймером."""
        if self._driver_result == "ok":
            self.status = MissionStatus.DONE
        else:
            self.status = MissionStatus.FAILED
            print(f"[РОБОТ] Миссия провалена: {self._driver_result}")
        return self.status

#временный_вывод

    def _run_driver(self):
        try:
            self.driver.execute(self.command_file)
            self._driver_result = "ok"
        except Exception as e:
            self._driver_result = f"error: {e}"