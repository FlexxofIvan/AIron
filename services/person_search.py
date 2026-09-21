# services/person_search.py
"""
PersonSearchMission — поиск человека через турель, наводка, запись mp4.

Оркестратор поверх:
  RobotDriver           — сессия, команды движения
  vision.FrameSource    — приём H.264-чанков
  vision.H264Decoder    — декодирование в BGR
  vision.YoloxDetector  — детекция людей
  vision.selection      — выбор цели и коррекции
  vision.VideoRecorder  — запись mp4

Живёт в отдельном потоке: UI дёргает start(), затем polls is_finished().
"""
import os
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

import numpy as np

from vision.calibration import (
    WARMUP_SEC, SWEEP_SEC, SWEEP_DIR, POST_SWEEP_SETTLE_SEC,
    ANALYZE_STRIDE,
    RETURN_TOLERANCE_SEC, RETURN_SETTLE_SEC,
    HORIZ_DEAD_ZONE_PX, HORIZ_STEP_SEC,
    VERT_EDGE_MARGIN_PX, VERT_STEP_SEC,
    FINE_MAX_ITERS, FINE_SETTLE_SEC,
    RECORD_SEC, RECORD_FPS, RECORDINGS_DIR,
)
from vision.detector import YoloxDetector
from vision.frame_source import FrameSource
from vision.h264_decoder import H264Decoder
from vision.recorder import VideoRecorder
from vision.selection import (
    pick_closest_person, score_detection,
    horizontal_correction, vertical_correction,
)
from vision.types import Detection


class SearchStatus(Enum):
    IDLE    = "idle"
    RUNNING = "running"
    DONE    = "done"
    FAILED  = "failed"


@dataclass
class SweepSample:
    ts_ms: int
    t_rel_sec: float
    det: Detection
    score: float


class PersonSearchMission:
    def __init__(self, driver, detector: YoloxDetector,
                 recordings_dir: str = RECORDINGS_DIR,
                 session_id: Optional[str] = None):
        self.driver = driver
        self.detector = detector
        self.recordings_dir = recordings_dir
        self.session_id = session_id or time.strftime("%Y%m%d_%H%M%S")

        self.status = SearchStatus.IDLE
        self.fs = FrameSource(maxsize=64)
        self.dec = H264Decoder()
        self._thread: Optional[threading.Thread] = None
        self._started_at: Optional[float] = None
        self._result: Optional[dict] = None
        self._error: Optional[str] = None

    # --- публичный API ---
    def start(self) -> None:
        if self.status == SearchStatus.RUNNING:
            raise RuntimeError("[РОБОТ] PersonSearch уже запущен")
        self.status = SearchStatus.RUNNING
        self._started_at = time.monotonic()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def is_finished(self) -> bool:
        return self.status in (SearchStatus.DONE, SearchStatus.FAILED)

    def elapsed(self) -> float:
        return 0.0 if self._started_at is None \
            else time.monotonic() - self._started_at

    @property
    def result(self) -> Optional[dict]:
        return self._result

    @property
    def error(self) -> Optional[str]:
        return self._error

    # --- основной поток ---
    def _run(self) -> None:
        try:
            self._do_mission()
            self.status = SearchStatus.DONE
        except Exception as e:
            self._error = str(e)
            self.status = SearchStatus.FAILED
            print(f"[РОБОТ] PersonSearch FAILED: {e}")
        finally:
            try:
                self.dec.close()
            except Exception:
                pass
            try:
                self.fs.close()
            except Exception:
                pass

    def _do_mission(self) -> None:
        print("[РОБОТ] PersonSearch: открываю сессию")
        self.driver.open_session(frame_callback=self.fs.on_chunk)
        session = self.driver.session()
        try:
            print("[РОБОТ] PersonSearch: turret + warmup")
            session.useTurretCamera()
            time.sleep(WARMUP_SEC)
            self._drain()

            print(f"[РОБОТ] PersonSearch: sweep {SWEEP_SEC:.1f} с, "
                  f"dir={SWEEP_DIR:+d}")
            samples = self._do_sweep(session)

            if not samples:
                print("[РОБОТ] PersonSearch: человек не найден")
                self._result = {"found": False, "recording": None}
                return

            best = max(samples, key=lambda s: s.score)
            print(f"[РОБОТ] PersonSearch: лучший кадр t={best.t_rel_sec:.2f} с, "
                  f"score={best.score:.4f}")

            self._return_to_best(session, best)
            self._fine_aim(session)
            path = self._record(session, RECORD_SEC)

            self._result = {
                "found": True,
                "recording": path,
                "best_ts_ms": best.ts_ms,
                "best_score": best.score,
            }
        finally:
            try:
                session.moveCameraHorizontal(0)
                session.moveCameraVertical(0)
            except Exception:
                pass
            self.driver.close_session()

    # --- helpers ---
    def _drain(self) -> None:
        while self.fs.get(timeout=0.01) is not None:
            pass

    def _grab_fresh_frame(self, discard: int = 3,
                          timeout_sec: float = 2.5) -> Optional[np.ndarray]:
        """Отсмотреть discard декодированных кадров, вернуть следующий."""
        seen = 0
        t0 = time.time()
        while (time.time() - t0) < timeout_sec:
            chunk = self.fs.get(timeout=0.3)
            if chunk is None:
                continue
            for img in self.dec.feed(chunk.data):
                seen += 1
                if seen > discard:
                    return img
        return None

    def _do_sweep(self, session) -> List[SweepSample]:
        self.fs.clear()
        try:
            session.moveCameraHorizontal(0)
        except Exception:
            pass
        time.sleep(0.2)
        self.fs.clear()

        samples: List[SweepSample] = []
        frame_idx = 0
        t_start = time.monotonic()
        session.moveCameraHorizontal(SWEEP_DIR)
        try:
            while (time.monotonic() - t_start) < SWEEP_SEC:
                chunk = self.fs.get(timeout=0.2)
                if chunk is None:
                    continue
                for img in self.dec.feed(chunk.data):
                    frame_idx += 1
                    if frame_idx % ANALYZE_STRIDE != 0:
                        continue
                    t_rel = time.monotonic() - t_start
                    dets = self.detector.detect_persons(img)
                    if not dets:
                        continue
                    h, w = img.shape[:2]
                    det = pick_closest_person(dets, w, h)
                    if det is None:
                        continue
                    samples.append(SweepSample(
                        ts_ms=chunk.ts_ms,
                        t_rel_sec=t_rel,
                        det=det,
                        score=score_detection(det, w, h),
                    ))
        finally:
            session.moveCameraHorizontal(0)
            time.sleep(POST_SWEEP_SETTLE_SEC)
        print(f"[РОБОТ] PersonSearch: sweep готов, с человеком кадров: "
              f"{len(samples)}")
        return samples

    def _return_to_best(self, session, best: SweepSample) -> None:
        if best.t_rel_sec < RETURN_TOLERANCE_SEC:
            print(f"[РОБОТ] PersonSearch: return skip "
                  f"(t={best.t_rel_sec:.2f})")
            return
        back_dir = -SWEEP_DIR
        print(f"[РОБОТ] PersonSearch: return {best.t_rel_sec:.2f} с, "
              f"dir={back_dir:+d}")
        session.moveCameraHorizontal(back_dir)
        time.sleep(best.t_rel_sec)
        session.moveCameraHorizontal(0)
        time.sleep(RETURN_SETTLE_SEC)
        self._drain()

    def _fine_aim(self, session) -> None:
        print("[РОБОТ] PersonSearch: fine aim")
        for it in range(FINE_MAX_ITERS):
            img = self._grab_fresh_frame(discard=3, timeout_sec=2.0)
            if img is None:
                print("[РОБОТ] PersonSearch: fine aim — нет кадра")
                return
            h, w = img.shape[:2]
            dets = self.detector.detect_persons(img)
            det = pick_closest_person(dets, w, h)
            if det is None:
                print("[РОБОТ] PersonSearch: цель потеряна")
                return

            v = vertical_correction(det, h, VERT_EDGE_MARGIN_PX)
            h_dir = horizontal_correction(det, w, HORIZ_DEAD_ZONE_PX)
            print(f"[РОБОТ] fine[{it:02d}] cx={det.center_x} "
                  f"y=[{det.y_min},{det.y_max}] -> H={h_dir:+d} V={v:+d}")

            if v == 0 and h_dir == 0:
                print("[РОБОТ] PersonSearch: цель в мёртвой зоне")
                return

            if v != 0:
                session.moveCameraVertical(v)
                time.sleep(VERT_STEP_SEC)
                session.moveCameraVertical(0)
                time.sleep(FINE_SETTLE_SEC)
            if h_dir != 0:
                session.moveCameraHorizontal(h_dir)
                time.sleep(HORIZ_STEP_SEC)
                session.moveCameraHorizontal(0)
                time.sleep(FINE_SETTLE_SEC)
            self._drain()

        print("[РОБОТ] PersonSearch: fine aim — исчерпаны итерации")

    def _record(self, session, seconds: float) -> str:
        path = os.path.join(self.recordings_dir, f"{self.session_id}.mp4")
        print(f"[РОБОТ] PersonSearch: запись {seconds:.0f} с -> {path}")
        rec = VideoRecorder(path, fps=RECORD_FPS, size=(640, 480))
        self.fs.clear()
        t_end = time.monotonic() + seconds
        try:
            while time.monotonic() < t_end:
                chunk = self.fs.get(timeout=0.2)
                if chunk is None:
                    continue
                for img in self.dec.feed(chunk.data):
                    rec.write(img)
        finally:
            rec.close()
        print(f"[РОБОТ] PersonSearch: записано {rec.frames_written} кадров, "
              f"~{rec.elapsed_sec:.1f} с")
        return path