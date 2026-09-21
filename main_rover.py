#!/usr/bin/env python
# -*- coding: utf-8 -*-

FRAMERATE = 20
DELAY_SEC = 2.5
AUDIO_RATE = 8192
AUDIO_CHANNELS = 1
AUDIO_OUTFILE = "audio.raw"
AUDIO_SAMPLE_BYTES = 2  # int16

from rover import Revolution

import json
import time
import sys
import signal
import subprocess
import tempfile
import os
import threading
import struct
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
COMMANDS_DIR = BASE_DIR / "commands"

try:
    import array as _array
    _ARRAY_TYPES = (_array.array,)
except Exception:
    _ARRAY_TYPES = ()

rover = None

def _signal_handler(sig, frame):
    global rover
    if rover is not None:
        rover.shutdown()
    sys.exit(0)

class FileRover(Revolution):

    def __init__(self, tmpfile, commands, audio_outfile, frame_callback=None):

        self.tmpfile = tmpfile
        self.commands = list(commands)
        self.cmd_index = 0
        self._stop_flag = threading.Event()
        self._cmd_thread = None
        self._frame_callback = frame_callback          # <-- NEW

        # Аудиофайл открываем сразу — сюда пишем PCM s16le
        self._audio_fh = open(audio_outfile, "wb", buffering=0)
        self._audio_lock = threading.Lock()
        self._audio_bytes_written = 0
        self._audio_samples_written = 0
        self._audio_outfile = audio_outfile
        self._closed = False

        super().__init__()

        # Поток команд — только если есть команды.
        # Для сессии (живое управление) commands == [] и поток не нужен.
        if self.commands:                              # <-- NEW
            self._cmd_thread = threading.Thread(
                target=self._command_loop, daemon=True
            )
            self._cmd_thread.start()

    def processVideo(self, imgbytes, timestamp_msec):
        # Сначала отдаём кадр наружу (для CV), потом пишем в tmp.
        # Исключение в callback НЕ должно валить reader-thread робота.
        if self._frame_callback is not None:
            try:
                self._frame_callback(imgbytes, timestamp_msec)
            except Exception as e:
                print(f"[main_rover] frame_callback error: {e}",
                      file=sys.stderr)
        try:
            self.tmpfile.write(imgbytes)
        except Exception:
            pass
    def processAudio(self, audiobytes, timestamp_msec):
        if self._closed:
            return
        data = self._to_pcm_bytes(audiobytes)
        if not data:
            return
        with self._audio_lock:
            if self._closed:
                return
            try:
                # (6) Вставка тишины, если пропущен блок аудио
                expected_samples = (timestamp_msec * AUDIO_RATE) // 1000
                gap_samples = expected_samples - self._audio_samples_written
                if gap_samples > 0:
                    self._audio_fh.write(
                        b'\x00' * (gap_samples * AUDIO_SAMPLE_BYTES)
                    )
                    self._audio_samples_written += gap_samples
                    self._audio_bytes_written += (
                        gap_samples * AUDIO_SAMPLE_BYTES
                    )
                self._audio_fh.write(data)
                n = len(data) // AUDIO_SAMPLE_BYTES
                self._audio_samples_written += n
                self._audio_bytes_written += len(data)
            except Exception:
                pass

    @staticmethod
    def _to_pcm_bytes(audiobytes):
        # (1) Поддержка bytes/bytearray/array/list/tuple
        if isinstance(audiobytes, (bytes, bytearray)):
            return bytes(audiobytes)
        if _ARRAY_TYPES and isinstance(audiobytes, _ARRAY_TYPES):
            return audiobytes.tobytes()
        if isinstance(audiobytes, (list, tuple)):
            try:
                return struct.pack(
                    '<%dh' % len(audiobytes),
                    *(int(x) for x in audiobytes)
                )
            except (struct.error, TypeError, ValueError):
                return b''
        return b''

    def _command_loop(self):
        for cmd in self.commands:
            if self._stop_flag.is_set():
                break
            try:
                self.execute(cmd)
            except Exception as e:
                print("Command error:", e, file=sys.stderr)
                break

            # (5) Безопасное чтение duration
            try:
                duration = float(cmd.get("duration") or 0.0)
            except (TypeError, ValueError):
                duration = 0.0
            if duration > 0 and self._stop_flag.wait(duration):
                break

        self._stop_motors()

    def execute(self, cmd):
        action = cmd["action"]

        if action == "drive":
            self.drive(
                int(cmd.get("wheel") or 0),
                int(cmd.get("steer") or 0),
                bool(cmd.get("slow", True)),
            )
        elif action == "stop":
            self.drive(0, 0, True)
        elif action == "stealth":
            (self.turnStealthOn if cmd.get("on", False)
             else self.turnStealthOff)()
        elif action == "turret":
            (self.useTurretCamera if cmd.get("on", False)
             else self.useDrivingCamera)()
        elif action == "camera-h":
            self.moveCameraHorizontal(int(cmd.get("dir") or 0))
        elif action == "camera-v":
            self.moveCameraVertical(int(cmd.get("dir") or 0))
        else:
            print("Unknown action:", action, file=sys.stderr)

    def _stop_motors(self):
        try:
            self.drive(0, 0, True)
            self.moveCameraHorizontal(0)
            self.moveCameraVertical(0)
        except Exception:
            pass

    def shutdown(self):
        if self._closed:
            return
        self._closed = True  

        self._stop_flag.set()
        self._stop_motors()

        # (A) Аккуратно выводим reader-поток и только потом
        #     закрываем аудиофайл
        try:
            self.is_active = False
        except Exception:
            pass
        try:
            if getattr(self, 'mediasock', None) is not None:
                self.mediasock.close()
        except Exception:
            pass
        rt = getattr(self, 'reader_thread', None)
        if rt is not None:
            try:
                rt.join(timeout=1.0)
            except Exception:
                pass

        with self._audio_lock:
            try:
                self._audio_fh.close()
            except Exception:
                pass

        try:
            self.close()
        except Exception:
            pass

        # Подсказка, как собрать WAV из сырого PCM
        dur = (self._audio_bytes_written
               / float(AUDIO_RATE * AUDIO_CHANNELS * AUDIO_SAMPLE_BYTES))
        print("Audio written: %s (%d bytes, ~%.1f sec)"
              % (self._audio_outfile, self._audio_bytes_written, dur))
        print("Convert to WAV with:")
        print("  ffmpeg -f s16le -ar %d -ac %d -i %s %s.wav"
              % (AUDIO_RATE, AUDIO_CHANNELS, self._audio_outfile,
                 self._audio_outfile))

class RoverSession:
    """Живая сессия с роботом вне JSON-команд.

    Открывается через open_session(). Пока жива — можно дёргать
    drive / moveCameraHorizontal / useTurretCamera и т.д. напрямую.
    Вызывающий обязан закрыть её через close().
    """

    def __init__(self, rover: "FileRover", tmpfile):
        self._rover = rover
        self._tmpfile = tmpfile

    # --- прокси публичных методов Rover / Revolution ---
    def drive(self, *a, **kw):
        return self._rover.drive(*a, **kw)

    def moveCameraHorizontal(self, *a, **kw):
        return self._rover.moveCameraHorizontal(*a, **kw)

    def moveCameraVertical(self, *a, **kw):
        return self._rover.moveCameraVertical(*a, **kw)

    def useTurretCamera(self, *a, **kw):
        return self._rover.useTurretCamera(*a, **kw)

    def useDrivingCamera(self, *a, **kw):
        return self._rover.useDrivingCamera(*a, **kw)

    def turnStealthOn(self, *a, **kw):
        return self._rover.turnStealthOn(*a, **kw)

    def turnStealthOff(self, *a, **kw):
        return self._rover.turnStealthOff(*a, **kw)

    def close(self):
        self._rover.shutdown()


def open_session(audio_outfile=AUDIO_OUTFILE, frame_callback=None,
                 startup_wait_sec=DELAY_SEC):
    """Открывает живую сессию с роботом без командного файла.

    Возвращает RoverSession. Бросает исключение, если робот недоступен.
    Вызывающий отвечает за session.close().
    """
    tmpfile = tempfile.NamedTemporaryFile()
    fr = FileRover(tmpfile, [], audio_outfile,
                   frame_callback=frame_callback)
    # Даём медиапотоку наполниться — reader-thread уже стартовал,
    # но первым кадрам нужно дойти до socket.
    time.sleep(startup_wait_sec)
    return RoverSession(fr, tmpfile)

# main --------------------------------------------------------------------

def _resolve_commands_file(commands_file):
    p = Path(commands_file)
    if p.is_absolute() or p.exists():
        if not p.exists():
            raise FileNotFoundError(f"Файл команд не найден: {p}")
        return p
    candidate = COMMANDS_DIR / p
    if candidate.exists():
        return candidate
    raise FileNotFoundError(
        f"Файл команд не найден: {commands_file}\n"
        f"Искал: {p} и {candidate}"
    )

def run(commands_file, audio_outfile=AUDIO_OUTFILE, register_sigint=True):
    global rover
    try:
        path = _resolve_commands_file(commands_file)
        with open(path, "r", encoding="utf-8") as f:
            commands = json.load(f)

        tmpfile = tempfile.NamedTemporaryFile()
        rover = FileRover(tmpfile, commands, audio_outfile,
                         frame_callback=frame_callback)

        if register_sigint:
            signal.signal(signal.SIGINT, _signal_handler)

        time.sleep(DELAY_SEC)

        # ffplay — опциональный просмотр видео.
        # Отсутствие в PATH не прерывает миссию.
        proc = None
        FNULL = None
        try:
            cmd = ['ffplay', '-window_title', 'Rover_Revolution',
                   '-framerate', str(FRAMERATE), tmpfile.name]
            FNULL = open(os.devnull, 'w')
            proc = subprocess.Popen(cmd, stdout=FNULL,
                                    stderr=subprocess.STDOUT)
        except FileNotFoundError:
            print("[main_rover] ffplay не найден — видео пропускаем.",
                  file=sys.stderr)

        # Истинный критерий окончания миссии — завершение команд FileRover.
        # ffplay может закрыться раньше, это не повод останавливать робота.
        try:
            if rover is not None and getattr(rover, "_cmd_thread", None) is not None:
                rover._cmd_thread.join()
        except KeyboardInterrupt:
            pass
        finally:
            if proc is not None:
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=2)
                    except Exception:
                        proc.kill()
                if FNULL is not None:
                    FNULL.close()
        return True
    except Exception as e:
        print(f"[main_rover] Ошибка: {e}", file=sys.stderr)
        return False
    finally:
        if rover is not None:
            try:
                rover.shutdown()
            except Exception:
                pass
            rover = None

if __name__ == '__main__':

    name = sys.argv[1] if len(sys.argv) > 1 else "commands.json"
    sys.exit(0 if run(name) else 1)



    