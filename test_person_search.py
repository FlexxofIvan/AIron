# test_person_search.py
"""
Запуск PersonSearchMission вне UI.
Драйвер открывает сессию, миссия ищет человека, наводится, пишет mp4.

Запуск:  python test_person_search.py
"""
import sys
import time

from services.robot_driver import RobotDriver
from services.person_search import PersonSearchMission
from vision.detector import YoloxDetector

import main_rover


ONNX_PATH = "yolox_nano.onnx"


def main():
    print("[TEST] Загружаю YOLOX...")
    det = YoloxDetector(ONNX_PATH, conf_threshold=0.3)

    driver = RobotDriver(main_rover.run, main_rover.open_session)
    mission = PersonSearchMission(driver=driver, detector=det)
    print(f"[TEST] Миссия: {mission.session_id}")

    mission.start()
    try:
        while not mission.is_finished():
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("[TEST] Ctrl+C, ждём завершения...")

    print(f"\n[TEST] Статус:  {mission.status}")
    print(f"[TEST] Ошибка:  {mission.error}")
    print(f"[TEST] Результат: {mission.result}")
    return 0 if mission.status.value == "done" else 1


if __name__ == "__main__":
    sys.exit(main())