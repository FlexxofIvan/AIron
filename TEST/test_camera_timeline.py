# test_camera_timeline.py
"""
30 секунд без команд. Снимает по одному кадру в секунду.
Покажет, меняется ли камера со временем.
"""
import os, sys, time
import cv2
from vision.frame_source import FrameSource
from vision.h264_decoder import H264Decoder

OUT_DIR = "camera_timeline"
DURATION_SEC = 30.0


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    fs = FrameSource(maxsize=64)
    dec = H264Decoder()

    print("[TEST] Открываю сессию, НИЧЕГО не вызываю...")
    import main_rover
    session = main_rover.open_session(frame_callback=fs.on_chunk)

    t_start = time.time()
    next_capture = 0.0
    saved = 0

    while (time.time() - t_start) < DURATION_SEC:
        chunk = fs.get(timeout=0.5)
        if chunk is None:
            continue
        for img in dec.feed(chunk.data):
            now = time.time() - t_start
            if now >= next_capture:
                path = f"{OUT_DIR}/t{now:05.1f}s_{saved:03d}.png"
                cv2.imwrite(path, img)
                print(f"  [{now:5.1f}s] {path}")
                saved += 1
                next_capture = now + 1.0

    print(f"\n[TEST] Сохранено {saved} кадров за {DURATION_SEC} с.")
    session.close()
    dec.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())