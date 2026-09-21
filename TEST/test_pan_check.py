# test_pan_check.py
"""
Проверка: панорамируется ли турель через moveCameraHorizontal?
1. Открываем сессию.
2. useTurretCamera() + прогрев 3 сек (переключение + экспозиция).
3. Кадр A.
4. moveCameraHorizontal(+1) на 1 сек → стоп → пауза → кадр B.
5. moveCameraHorizontal(-1) на 1 сек → стоп → пауза → кадр C (должны вернуться к A).
"""
import os, sys, time
import cv2
from vision.frame_source import FrameSource
from vision.h264_decoder import H264Decoder

OUT = "pan_check"
SWITCH_WARMUP_SEC = 3.0   # переключение камеры + экспозиция
PAN_SEC = 1.0
SETTLE_SEC = 0.7


def grab_one(fs, dec, timeout=3.0):
    t = time.time()
    while (time.time() - t) < timeout:
        chunk = fs.get(timeout=0.3)
        if chunk is None:
            continue
        frames = dec.feed(chunk.data)
        if frames:
            return frames[-1]
    return None


def save(fs, dec, name):
    img = grab_one(fs, dec)
    if img is None:
        print(f"[TEST] Не дождался кадра для {name}", file=sys.stderr)
        return False
    cv2.imwrite(f"{OUT}/{name}.png", img)
    print(f"[TEST] {name}.png сохранён")
    return True


def main():
    os.makedirs(OUT, exist_ok=True)
    fs = FrameSource(maxsize=64)
    dec = H264Decoder()

    import main_rover
    session = main_rover.open_session(frame_callback=fs.on_chunk)

    print("[TEST] Переключаюсь на турель...")
    session.useTurretCamera()
    print(f"[TEST] Прогрев {SWITCH_WARMUP_SEC} с...")
    time.sleep(SWITCH_WARMUP_SEC)
    fs.clear()

    print("[TEST] Кадр A — исходный...")
    if not save(fs, dec, "A_before"):
        session.close(); dec.close(); return 1

    print(f"[TEST] moveCameraHorizontal(+1) на {PAN_SEC} с...")
    session.moveCameraHorizontal(+1)
    time.sleep(PAN_SEC)
    session.moveCameraHorizontal(0)
    time.sleep(SETTLE_SEC)
    fs.clear()

    print("[TEST] Кадр B — после +1...")
    if not save(fs, dec, "B_pan_right"):
        session.close(); dec.close(); return 1

    print(f"[TEST] moveCameraHorizontal(-1) на {PAN_SEC} с...")
    session.moveCameraHorizontal(-1)
    time.sleep(PAN_SEC)
    session.moveCameraHorizontal(0)
    time.sleep(SETTLE_SEC)
    fs.clear()

    print("[TEST] Кадр C — после возврата -1...")
    if not save(fs, dec, "C_pan_back"):
        session.close(); dec.close(); return 1

    print("\n[TEST] Готово. Сравни глазами:")
    print("  A vs B — сцена повернулась вправо?")
    print("  A vs C — сцена вернулась примерно туда же?")

    session.close()
    dec.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())