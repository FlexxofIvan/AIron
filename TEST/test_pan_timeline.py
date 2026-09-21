# test_pan_timeline.py
"""
Таймлайн: записываем кадры с турели каждые 0.4 с в течение 10 с.
Панорамирование включается на 2-й секунде, выключается на 6-й.
Открываем последовательность — видно, движется ли картинка.
"""
import os, sys, time
import cv2
from vision.frame_source import FrameSource
from vision.h264_decoder import H264Decoder

OUT = "pan_timeline"
SWITCH_WARMUP_SEC = 3.0
DURATION_SEC = 10.0
CAPTURE_EVERY_SEC = 0.4
PAN_START_SEC = 2.0
PAN_STOP_SEC = 6.0
PAN_DIR = +1


def main():
    os.makedirs(OUT, exist_ok=True)
    fs = FrameSource(maxsize=64)
    dec = H264Decoder()

    import main_rover
    session = main_rover.open_session(frame_callback=fs.on_chunk)

    print("[TEST] useTurretCamera()...")
    session.useTurretCamera()
    print(f"[TEST] Прогрев {SWITCH_WARMUP_SEC} с...")
    time.sleep(SWITCH_WARMUP_SEC)
    fs.clear()

    # Сброс возможного залипшего isMoving у _RoverCamera
    try:
        session.moveCameraHorizontal(0)
    except Exception:
        pass
    time.sleep(0.3)
    fs.clear()

    print(f"[TEST] Съёмка {DURATION_SEC} с. "
          f"Пан с {PAN_START_SEC} по {PAN_STOP_SEC} с...")

    t0 = time.time()
    next_capture = 0.0
    saved = 0
    pan_started = False
    pan_stopped = False

    while (time.time() - t0) < DURATION_SEC:
        elapsed = time.time() - t0

        if elapsed >= PAN_START_SEC and not pan_started:
            print(f"  [{elapsed:5.1f}s] START pan dir={PAN_DIR}")
            session.moveCameraHorizontal(PAN_DIR)
            pan_started = True
        if elapsed >= PAN_STOP_SEC and not pan_stopped:
            print(f"  [{elapsed:5.1f}s] STOP pan")
            session.moveCameraHorizontal(0)
            pan_stopped = True

        chunk = fs.get(timeout=0.2)
        if chunk is None:
            continue
        for img in dec.feed(chunk.data):
            if elapsed >= next_capture:
                path = f"{OUT}/t{elapsed:05.1f}s_{saved:03d}.png"
                cv2.imwrite(path, img)
                saved += 1
                next_capture = elapsed + CAPTURE_EVERY_SEC

    print(f"\n[TEST] Сохранено {saved} кадров в {OUT}/")
    session.close()
    dec.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())