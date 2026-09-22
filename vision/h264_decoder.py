# vision/h264_decoder.py
"""
Обёртка над PyAV для декодирования H.264-потока с робота.

Один экземпляр на сессию. Внутренний CodecContext хранит состояние
между вызовами feed() — SPS/PPS и опорные кадры не теряются.

Не потокобезопасно на уровне одновременных feed(): защищаем lock'ом
на случай, если кто-то будет звать из разных потоков. Но по дизайну
feed() должен зваться из одного decoder-thread.
"""
import threading

import av
import numpy as np


class H264Decoder:
    def __init__(self):
        # "r" — reader mode. PyAV сам подхватит SPS/PPS из потока.
        self._codec = av.CodecContext.create("h264", "r")
        self._lock = threading.Lock()
        self._frames_out = 0
        self._chunks_in = 0

    def feed(self, chunk: bytes):
        """Принять очередной кусок Annex-B H.264.

        Возвращает список numpy BGR-кадров (uint8, H×W×3).
        Может быть пусто — decoder не всегда выдаёт кадр на каждый chunk,
        накапливает состояние.
        """
        out = []
        with self._lock:
            self._chunks_in += 1
            try:
                for packet in self._codec.parse(chunk):
                    for frame in self._codec.decode(packet):
                        img = frame.to_ndarray(format="bgr24")
                        out.append(np.ascontiguousarray(img))
                        self._frames_out += 1
            except Exception as e:
                print(f"[H264] decode error: {e}")
        return out

    def close(self):
        with self._lock:
            try:
                self._codec.close()
            except Exception:
                pass

    @property
    def stats(self):
        return {
            "chunks_in": self._chunks_in,
            "frames_out": self._frames_out,
        }