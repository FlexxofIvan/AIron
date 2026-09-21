# vision/recorder.py
"""
Пишет поток BGR-кадров в mp4 через cv2.VideoWriter.
Не знает про робота — принимает готовые кадры.
Путь приходит аргументом; вызывающий решает, куда писать.
"""
import os
import time

import cv2
import numpy as np


class VideoRecorder:
    def __init__(self, path: str, fps: float, size=(640, 480),
                 fourcc: str = "mp4v"):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.path = path
        self.fps = float(fps)
        self.size = (int(size[0]), int(size[1]))
        self._writer = cv2.VideoWriter(
            path, cv2.VideoWriter_fourcc(*fourcc), self.fps, self.size
        )
        if not self._writer.isOpened():
            raise RuntimeError(f"[REC] cv2.VideoWriter не открылся: {path}")
        self._frames = 0
        self._closed = False
        self._t0 = time.time()

    def write(self, frame_bgr: np.ndarray) -> None:
        if self._closed:
            return
        h, w = frame_bgr.shape[:2]
        if (w, h) != self.size:
            frame_bgr = cv2.resize(frame_bgr, self.size,
                                   interpolation=cv2.INTER_LINEAR)
        self._writer.write(frame_bgr)
        self._frames += 1

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._writer.release()
        except Exception:
            pass

    @property
    def frames_written(self) -> int:
        return self._frames

    @property
    def elapsed_sec(self) -> float:
        return time.time() - self._t0