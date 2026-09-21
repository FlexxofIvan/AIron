# test_camera_default.py — минимальный, на 20 секунд
import os, sys, time
import cv2
from vision.frame_source import FrameSource
from vision.h264_decoder import H264Decoder

os.makedirs("camera_default", exist_ok=True)
fs = FrameSource(maxsize=64)
dec = H264Decoder()

import main_rover
session = main_rover.open_session(frame_callback=fs.on_chunk)
print("[TEST] Сессия открыта. НИЧЕГО не вызываю. Собираю 20 сек...")

t_end = time.time() + 20.0
saved = 0
while time.time() < t_end:
    chunk = fs.get(timeout=0.5)
    if chunk is None:
        continue
    for img in dec.feed(chunk.data):
        if saved % 10 == 0:   # каждый 10-й, чтобы не заваливать диск
            path = f"camera_default/t{int(time.time()-t_end+20):02d}_{saved:03d}.png"
            cv2.imwrite(path, img)
            print(f"  [{saved:03d}] {path}")
        saved += 1

print(f"[TEST] Всего декодировано: {saved} кадров за 20 с")
print(f"[TEST] Ожидаемо ~200-400 (если поток ~10-20 fps)")
session.close()
dec.close()