"""Camera, recording and snapshots for Nirni Livewebcam.

Copyright (c) 2026 Sidagam Nirni Naidu. All rights reserved.
Use of this file is governed by the LICENSE file in this folder.

CameraStream  reads the webcam on its own thread and always hands out the
              newest picture, so the rest of the program never waits on the camera
Recorder      writes a video file on its own thread, at a steady frame rate
save_snapshot saves one picture as PNG

Pictures and videos are saved only when you ask for them, only on this
computer, in the folder given by --output-dir (default: captures/).
"""

from __future__ import annotations

import queue
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


class CameraStream:
    """Background reader. read() returns (picture, id) for the newest picture
    that is newer than last_id, or None if there is none yet."""

    FAIL_SECONDS = 3.0

    def __init__(self, capture, mirror: bool = True):
        self.capture = capture
        self.mirror = mirror
        self.error: str | None = None
        self._frame = None
        self._id = 0
        self._cond = threading.Condition()
        self._stop = False
        self.measured_fps = 0.0
        self._thread = threading.Thread(target=self._run, name="nirni-camera", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        failing_since = None
        last = time.perf_counter()
        while not self._stop:
            ok, frame = self.capture.read()
            if not ok or frame is None:
                if failing_since is None:
                    failing_since = time.perf_counter()
                elif time.perf_counter() - failing_since > self.FAIL_SECONDS:
                    self.error = "The webcam stopped returning images. Is it still plugged in?"
                    with self._cond:
                        self._cond.notify_all()
                    return
                time.sleep(0.02)
                continue
            failing_since = None
            if self.mirror:
                frame = cv2.flip(frame, 1)
            now = time.perf_counter()
            self.measured_fps = 0.9 * self.measured_fps + 0.1 / max(now - last, 1e-3)
            last = now
            with self._cond:
                self._frame = frame
                self._id += 1
                self._cond.notify_all()

    def read(self, last_id: int = 0, timeout: float = 1.0):
        deadline = time.perf_counter() + timeout
        with self._cond:
            while self._id <= last_id and self.error is None and not self._stop:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    return None
                self._cond.wait(remaining)
            if self.error is not None:
                raise RuntimeError(self.error)
            if self._id <= last_id:
                return None
            return self._frame, self._id

    def close(self) -> None:
        self._stop = True
        with self._cond:
            self._cond.notify_all()
        self._thread.join(timeout=2.0)
        self.capture.release()


class FramePacer:
    """Says how many video frames are due at each moment, so the saved video
    plays at the right speed even when the program runs faster or slower."""

    def __init__(self, fps: float):
        self.step = 1.0 / fps
        self.next_time: float | None = None

    def due(self, now: float) -> int:
        if self.next_time is None:
            self.next_time = now + self.step
            return 1
        if now < self.next_time:
            return 0
        count = min(int((now - self.next_time) / self.step) + 1, 5)
        self.next_time += count * self.step
        if now - self.next_time > 1.0:          # fell far behind: restart the clock
            self.next_time = now + self.step
        return count


def stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def unique_path(directory: Path, suffix: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    base = f"nirni-{stamp()}"
    path = directory / f"{base}{suffix}"
    counter = 2
    while path.exists():
        path = directory / f"{base}-{counter}{suffix}"
        counter += 1
    return path


def save_snapshot(frame: np.ndarray, directory: Path) -> Path:
    """Save one picture as PNG. Raises RuntimeError if it cannot be written."""
    path = unique_path(Path(directory), ".png")
    if not cv2.imwrite(str(path), frame):
        raise RuntimeError(f"could not write {path}")
    return path


class Recorder:
    """Records the clean output (what your call sees, no preview hints).
    There is no sound: the program never touches the microphone."""

    CODECS = (("mp4v", ".mp4"), ("avc1", ".mp4"), ("MJPG", ".avi"))

    def __init__(self, directory: Path, fps: float = 30.0):
        self.directory = Path(directory)
        self.fps = fps
        self.recording = False
        self.path: Path | None = None
        self.frames = 0
        self.dropped = 0
        self.started_at = 0.0
        self._owed = 0
        self._last_copy = None
        self.size = (0, 0)
        self._pacer = FramePacer(fps)
        self._queue: queue.Queue = queue.Queue(maxsize=12)
        self._writer = None
        self._thread: threading.Thread | None = None

    def start(self, size: tuple[int, int], now: float) -> Path:
        if self.recording:
            return self.path
        width, height = size
        for codec, suffix in self.CODECS:
            path = unique_path(self.directory, suffix)
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), self.fps, (width, height))
            if writer.isOpened():
                break
            writer.release()
            path.unlink(missing_ok=True)
        else:
            raise RuntimeError("no video codec is available in this OpenCV build")
        self._writer, self.path = writer, path
        self.size = (width, height)
        self.frames = self.dropped = 0
        self._owed = 0
        self._last_copy = None
        self.started_at = now
        self._pacer = FramePacer(self.fps)
        self._queue = queue.Queue(maxsize=12)
        self._thread = threading.Thread(target=self._run, name="nirni-record", daemon=True)
        self.recording = True
        self._thread.start()
        return path

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            frame, count = item
            for _ in range(count):
                self._writer.write(frame)
                self.frames += 1

    def write(self, frame: np.ndarray, now: float) -> None:
        if not self.recording:
            return
        count = self._pacer.due(now)
        if not count:
            return
        if (frame.shape[1], frame.shape[0]) != self.size:      # the camera changed its picture size
            copy = cv2.resize(frame, self.size, interpolation=cv2.INTER_LINEAR)
        else:
            copy = frame.copy()
        try:
            # If the disk or encoder fell behind, the missed frames are repeated
            # here, so the video keeps its real length instead of speeding up.
            self._queue.put_nowait((copy, count + self._owed))
            self._owed = 0
            self._last_copy = copy
        except queue.Full:
            self._owed += count
            self.dropped += count

    def elapsed(self, now: float) -> float:
        return now - self.started_at if self.recording else 0.0

    def stop(self) -> tuple[Path | None, int]:
        if not self.recording:
            return None, 0
        self.recording = False
        if self._owed and self._last_copy is not None:
            self._queue.put((self._last_copy, self._owed))
            self._owed = 0
        self._queue.put(None)
        if self._thread is not None:
            self._thread.join(timeout=10.0)
        if self._writer is not None:
            self._writer.release()
        self._writer = None
        return self.path, self.frames
