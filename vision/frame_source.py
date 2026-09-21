# vision/frame_source.py
"""
FrameSource — thread-safe приёмник H.264-чанков из main_rover.

on_chunk() вызывается из reader-thread робота — он не должен блокироваться.
Внутри ограниченная очередь с drop-oldest: не успевает потребитель —
теряем старые чанки, не тормозим медиапоток.
"""
import queue
import threading
from typing import Optional

from .types import Chunk


class FrameSource:
    def __init__(self, maxsize: int = 30):
        # maxsize больше, чем для JPEG: H.264-чанки маленькие и их много
        self._q: "queue.Queue[Chunk]" = queue.Queue(maxsize=maxsize)
        self._lock = threading.Lock()
        self._closed = False
        self._dropped = 0

    def on_chunk(self, data: bytes, ts_ms: int) -> None:
        """Callback для FileRover.processVideo. Никогда не блокирует."""
        if self._closed:
            return
        chunk = Chunk(data=bytes(data), ts_ms=int(ts_ms))
        try:
            self._q.put_nowait(chunk)
        except queue.Full:
            try:
                self._q.get_nowait()
            except queue.Empty:
                pass
            try:
                self._q.put_nowait(chunk)
            except queue.Full:
                pass
            with self._lock:
                self._dropped += 1

    def get(self, timeout: float = 0.2) -> Optional[Chunk]:
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None

    def clear(self) -> None:
        with self._lock:
            while True:
                try:
                    self._q.get_nowait()
                except queue.Empty:
                    break

    def close(self) -> None:
        self._closed = True

    @property
    def dropped_count(self) -> int:
        with self._lock:
            return self._dropped