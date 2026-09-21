# test_camera_switch.py
"""
Диагностика переключения камеры.
Собирает 5 кадров до useTurretCamera() и 5 после — с паузой и
пересозданием декодера (новый H.264-поток = новые SPS/PPS).
"""
import os
import sys
import time
import hashlib

import cv2

from vision.frame_source import FrameSource
from vision.h264_decoder import H264Decoder


OUT_DIR = "camera_test"
WAIT_AFTER_SWITCH_SEC = 3.0


def collect(fs, decoder, n, prefix, timeout_sec=10.0):
    saved = 0
    t_start = time.time()
    while saved < n and (time.time() - t_start) < timeout_sec:
        chunk = fs.get(timeout=0.5)
        if chunk is None:
            continue
        for img in decoder.feed(chunk.data):
            path = os.path.join(OUT_DIR, f"{prefix}_{saved:03d}.png")
            cv2.imwrite(path, img)
            print(f"  {prefix}[{saved:02d}] {img.shape[1]}x{img.shape[0]}  "
                  f"ts={chunk.ts_ms}")
            saved += 1
            if saved >= n:
                break
    return saved


def file_hash(path):
    if not os.path.exists(path):
        return "-"
    return hashlib.md5(open(path, "rb").read()).hexdigest()[:8]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    fs = FrameSource(maxsize=64)
    decoder = H264Decoder()

    print("[TEST] Открываю сессию...")
    import main_rover
    try:
        session = main_rover.open_session(frame_callback=fs.on_chunk)
    except Exception as e:
        print(f"[TEST] Ошибка: {e}", file=sys.stderr)
        return 1

    print("[TEST] Собираю 5 кадров с ТЕКУЩЕЙ (по умолчанию) камеры...")
    collect(fs, decoder, 5, "before")

    print("[TEST] Отправляю useTurretCamera()...")
    session.useTurretCamera()

    print(f"[TEST] Жду {WAIT_AFTER_SWITCH_SEC} с и пересоздаю декодер...")
    time.sleep(WAIT_AFTER_SWITCH_SEC)
    fs.clear()
    decoder.close()
    decoder = H264Decoder()   # свежий контекст под новый поток

    print("[TEST] Собираю 5 кадров ПОСЛЕ переключения...")
    collect(fs, decoder, 5, "after")

    h1 = file_hash(os.path.join(OUT_DIR, "before_000.png"))
    h2 = file_hash(os.path.join(OUT_DIR, "after_000.png"))
    print(f"\n[TEST] before_000.png hash: {h1}")
    print(f"[TEST] after_000.png  hash: {h2}")

    if h1 == "-" or h2 == "-":
        print("[TEST] Один из наборов пустой — смотри выше по логу.")
    elif h1 == h2:
        print("[TEST] Кадры ИДЕНТИЧНЫ. Камера НЕ переключилась.")
    else:
        print("[TEST] Кадры РАЗНЫЕ. Переключение работает.")

    session.close()
    decoder.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())