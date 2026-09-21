# test_tilt_check.py — заменить целиком
"""
Три положения башни: A (исходное) → B (после +1 × 3 с) → C (после -1 × 3 с).
После каждого манёвра ПЕРЕСОЗДАЁМ декодер и скипаем N кадров —
это ключ, чтобы получить свежие кадры без буферизованной задержки.
"""
import os, sys, time
import cv2
from vision.frame_source import FrameSource
from vision.h264_decoder import H264Decoder

OUT = "tilt_check"
WARMUP_SEC = 3.0
TILT_SEC = 3.0
SETTLE_SEC = 1.0
DISCARD_FRAMES = 15   # сколько кадров слить после смены положения
SAVE_FRAMES = 3       # сколько сохранить для сравнения


def grab_fresh(fs, dec, discard_n):
    """Слить discard_n кадров и вернуть следующий."""
    seen = 0
    t = time.time()
    while (time.time() - t) < 6.0:
        ch = fs.get(timeout=0.3)
        if ch is None:
            continue
        frames = dec.feed(ch.data)
        for img in frames:
            seen += 1
            if seen > discard_n:
                return img
    return None


def save_burst(fs, dec, prefix, n=SAVE_FRAMES, discard=DISCARD_FRAMES):
    """Сохранить n свежих кадров подряд после сброса буфера."""
    print(f"[TEST] {prefix}: сливаю {discard} кадров, сохраняю {n}...")
    seen = 0
    saved = 0
    t = time.time()
    while saved < n and (time.time() - t) < 8.0:
        ch = fs.get(timeout=0.3)
        if ch is None:
            continue
        frames = dec.feed(ch.data)
        for img in frames:
            seen += 1
            if seen <= discard:
                continue
            path = f"{OUT}/{prefix}_{saved:02d}.png"
            cv2.imwrite(path, img)
            print(f"  -> {path}")
            saved += 1
            if saved >= n:
                break
    return saved


def reset_decoder(fs, dec):
    """Сбросить очередь и пересоздать декодер (новый H.264 контекст)."""
    fs.clear()
    dec.close()
    return H264Decoder()


def main():
    os.makedirs(OUT, exist_ok=True)
    fs = FrameSource(maxsize=64)
    dec = H264Decoder()

    import main_rover
    s = main_rover.open_session(frame_callback=fs.on_chunk)

    print("[TEST] useTurretCamera() + прогрев...")
    s.useTurretCamera()
    time.sleep(WARMUP_SEC)
    dec = reset_decoder(fs, dec)

    print("[TEST] === A: исходное положение ===")
    save_burst(fs, dec, "A")

    print(f"[TEST] moveCameraVertical(+1) на {TILT_SEC} с...")
    s.moveCameraVertical(+1)
    time.sleep(TILT_SEC)
    s.moveCameraVertical(0)
    time.sleep(SETTLE_SEC)
    dec = reset_decoder(fs, dec)

    print("[TEST] === B: после +1 ===")
    save_burst(fs, dec, "B")

    print(f"[TEST] moveCameraVertical(-1) на {TILT_SEC} с...")
    s.moveCameraVertical(-1)
    time.sleep(TILT_SEC)
    s.moveCameraVertical(0)
    time.sleep(SETTLE_SEC)
    dec = reset_decoder(fs, dec)

    print("[TEST] === C: после -1 ===")
    save_burst(fs, dec, "C")

    print("\n[TEST] Открой A_00, B_00, C_00 и сравни.")
    print("[TEST] Вопрос: B отличается от A? Куда наклонилось — вверх/вниз?")
    print("[TEST] И: C вернулось к A?")

    s.close()
    dec.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())