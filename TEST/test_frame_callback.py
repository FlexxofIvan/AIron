# test_frame_callback.py
"""
Одноразовый тест: доходят ли кадры из робота в Python и декодируются ли.

Сохраняет первые N декодированных кадров в frames_test/*.png и печатает
статистику: сколько чанков пришло, сколько кадров из них вышло, FPS.

Запуск:  python test_frame_callback.py
"""
import os
import sys
import time

import cv2

from vision.frame_source import FrameSource
from vision.h264_decoder import H264Decoder


OUT_DIR = "frames_test"
N_FRAMES = 10
TIMEOUT_SEC = 20.0


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    fs = FrameSource(maxsize=64)
    decoder = H264Decoder()

    print("[TEST] Открываю сессию с роботом...")
    import main_rover
    try:
        session = main_rover.open_session(frame_callback=fs.on_chunk)
    except Exception as e:
        print(f"[TEST] Не удалось открыть сессию: {e}", file=sys.stderr)
        return 1

    print("[TEST] Переключаюсь на turret-камеру...")
    try:
        session.useTurretCamera()
    except Exception as e:
        print(f"[TEST] useTurretCamera упал: {e}", file=sys.stderr)

    print(f"[TEST] Собираю {N_FRAMES} кадров (таймаут {TIMEOUT_SEC} с)...")
    saved = 0
    chunks_seen = 0
    first_ts = None
    last_ts = None
    t_start = time.time()

    while saved < N_FRAMES and (time.time() - t_start) < TIMEOUT_SEC:
        chunk = fs.get(timeout=0.5)
        if chunk is None:
            continue

        chunks_seen += 1
        if first_ts is None:
            first_ts = chunk.ts_ms

        frames = decoder.feed(chunk.data)
        for img in frames:
            last_ts = chunk.ts_ms
            path = os.path.join(OUT_DIR, f"frame_{saved:03d}.png")
            cv2.imwrite(path, img)
            print(f"  [{saved:02d}] {img.shape[1]}x{img.shape[0]}  "
                  f"ts={chunk.ts_ms}  -> {path}")
            saved += 1
            if saved >= N_FRAMES:
                break

    dt = (last_ts - first_ts) / 1000.0 if (first_ts and last_ts) else 0.0
    print(f"\n[TEST] Чанков получено:       {chunks_seen}")
    print(f"[TEST] Декодировано кадров:   {saved}")
    print(f"[TEST] Длительность потока:   {dt:.2f} с")
    print(f"[TEST] Дропнуто в очереди:    {fs.dropped_count}")
    print(f"[TEST] Статистика декодера:   {decoder.stats}")

    try:
        session.close()
    except Exception as e:
        print(f"[TEST] Ошибка при закрытии: {e}", file=sys.stderr)
    decoder.close()

    if saved == 0:
        print("[TEST] НИ ОДНОГО КАДРА. Пламбинг или декодер сломан.",
              file=sys.stderr)
        return 1

    print(f"\n[TEST] OK. Открой {OUT_DIR}/frame_000.png и проверь глазами.")
    return 0


if __name__ == "__main__":
    sys.exit(main())