# services/robot_driver.py
"""
Единственное место в проекте, где UI и сервисы знают про main_rover.run.
Позже здесь же появятся emergency_stop и реальный serial-протокол.
"""


class RobotDriver:
    def __init__(self, run_fn, open_session_fn):
        self._run_fn = run_fn
        self._open_session_fn = open_session_fn
        self._session = None

    def execute(self, command_file: str) -> None:
        # main_rover.run вызывается из фонового потока Mission,
        # поэтому SIGINT там регистрировать нельзя.
        result = self._run_fn(command_file, register_sigint=False)
        if result is False:
            raise RuntimeError(
                "main_rover.run вернул False (робот не выполнил команды)"
            )

    def open_session(self, frame_callback=None) -> None:
        if self._session is not None:
            raise RuntimeError("[ROBOT] Сессия уже открыта")
        self._session = self._open_session_fn(frame_callback=frame_callback)

    def session(self):
        """Возвращает RoverSession. Бросает, если сессия не открыта."""
        if self._session is None:
            raise RuntimeError("[ROBOT] Сессия не открыта")
        return self._session

    def close_session(self) -> None:
        if self._session is not None:
            try:
                self._session.close()
            finally:
                self._session = None

    def emergency_stop(self) -> None:
        print("[ROBOT] emergency_stop пока не реализован.")