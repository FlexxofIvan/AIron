# test_yolox_on_frames.py
"""
Прогон YOLOX-Nano на всех PNG из указанной папки.
Рисует рамки, сохраняет результат в <папка>/out_*.png, печатает
статистику (сколько детекций, среднее время инференса).

Запуск:
    python test_yolox_on_frames.py frames_test
    python test_yolox_on_frames.py pan_check
    python test_yolox_on_frames.py camera_test
"""
import os
import sys
import glob

import cv2

from vision.detector import YoloxDetector


ONNX_PATH = "yolox_nano.onnx"


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_yolox_on_frames.py <dir_with_png>")
        return 1

    src_dir = sys.argv[1]
    if not os.path.isdir(src_dir):
        print(f"[TEST] Нет папки: {src_dir}", file=sys.stderr)
        return 1

    pngs = sorted(glob.glob(os.path.join(src_dir, "*.png")))
    if not pngs:
        print(f"[TEST] В {src_dir} нет PNG", file=sys.stderr)
        return 1

    print(f"[TEST] Загружаю модель: {ONNX_PATH}")
    det = YoloxDetector(ONNX_PATH, conf_threshold=0.3)

    print(f"[TEST] Файлов: {len(pngs)}")
    total_dets = 0
    total_ms = 0.0

    for path in pngs:
        img = cv2.imread(path)
        if img is None:
            print(f"  [skip] {path} — не читается")
            continue

        import time
        t0 = time.time()
        dets = det.detect_persons(img)
        dt_ms = (time.time() - t0) * 1000.0
        total_ms += dt_ms
        total_dets += len(dets)

        for d in dets:
            cv2.rectangle(img, (d.x_min, d.y_min), (d.x_max, d.y_max),
                          (0, 255, 0), 2)
            cv2.putText(img, f"{d.score:.2f}",
                        (d.x_min, max(12, d.y_min - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        out_path = os.path.join(src_dir, "out_" + os.path.basename(path))
        cv2.imwrite(out_path, img)

        print(f"  {os.path.basename(path):30s} "
              f"dets={len(dets):2d}  {dt_ms:5.1f} ms  → {out_path}")

    n = len(pngs)
    print(f"\n[TEST] Всего кадров:       {n}")
    print(f"[TEST] Среднее время:      {total_ms/n:.1f} ms")
    print(f"[TEST] Детекций всего:     {total_dets}")
    print(f"[TEST] Открой out_*.png глазами — видны ли зелёные рамки?")
    return 0


if __name__ == "__main__":
    sys.exit(main())