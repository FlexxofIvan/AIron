# test_presweep_tilt.py
"""
Пять положений: дефолт, +0.5с, +1.0с, +1.5с, +2.0с вверх.
Сохраняет по 3 кадра на каждое.
Нужно встать в 1.5-2 м от робота лицом к нему.
"""
import os, sys, time
import cv2
from vision.frame_source import FrameSource
from vision.h264_decoder import H264Decoder

OUT = "presweep_test"
WARMUP = 3.0
SETTLE = 1.0
DISCARD = 10
SAVE = 3
STEPS = [0.0, 0.3, 0.6, 0.9, 1.0]


def save_burst(fs, dec, prefix):
    seen, saved = 0, 0
    t = time.time()
    while saved < SAVE and (time.time() - t) < 6.0:
        ch = fs.get(timeout=0.3)
        if ch is None:
            continue
        for img in dec.feed(ch.data):
            seen += 1
            if seen <= DISCARD:
                continue
            cv2.imwrite(f"{OUT}/{prefix}_{saved:02d}.png", img)
            saved += 1
            if saved >= SAVE:
                break
    print(f"  {prefix}: {saved} кадров")


def reset(fs, dec):
    fs.clear()
    dec.close()
    return H264Decoder()


def main():
    os.makedirs(OUT, exist_ok=True)
    fs = FrameSource(maxsize=64)
    dec = H264Decoder()
    import main_rover
    s = main_rover.open_session(frame_callback=fs.on_chunk)

    s.useTurretCamera()
    time.sleep(WARMUP)
    dec = reset(fs, dec)

    print("[TEST] === Дефолт ===")
    save_burst(fs, dec, "t00_0s")

    for i, up_sec in enumerate(STEPS[1:], start=1):
        print(f"[TEST] === +{up_sec:.1f} с вверх ===")
        s.moveCameraVertical(+1)
        time.sleep(up_sec)
        s.moveCameraVertical(0)
        time.sleep(SETTLE)
        dec = reset(fs, dec)
        save_burst(fs, dec, f"t{i:02d}_{up_sec:.1f}s")

    print("\n[TEST] Открой все tXX_00.png и скажи: на каком шаге "
          "человек виден от головы до пояса / до колен?")
    s.close(); dec.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())