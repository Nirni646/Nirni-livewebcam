"""Face and hand tracking for Nirni Livewebcam.

Copyright (c) 2026 Sidagam Nirni Naidu. All rights reserved.
Use of this file is governed by the LICENSE file in this folder.

Detector       runs the MediaPipe face and hand models on one picture
DetectionWorker  runs a Detector on its own thread and always works on the
               newest picture, so a slow detection never slows the video down
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from air_canvas import dedupe_hands

# Face mesh landmark numbers (MediaPipe canonical face model).
NOSE_TIP_IDS = (1, 4)
FOREHEAD_IDS = (10, 9)       # top of the forehead and between the eyebrows
CHIN_IDS = (152,)
MOUTH_IDS = (13, 14)         # inner lips, middle
FACE_LEFT_ID = 234
FACE_RIGHT_ID = 454

DETECT_WIDTH = 512           # pictures are shrunk to this width before the models see them


@dataclass
class Observation:
    """Everything the trackers saw in one picture. Positions are pixels of the full frame."""
    hands: list = field(default_factory=list)       # one (21, 2) array per hand
    face_found: bool = False
    nose: tuple[float, float] | None = None
    forehead: tuple[float, float] | None = None
    chin: tuple[float, float] | None = None
    mouth: tuple[float, float] | None = None
    face_width: float = 0.0
    blink_left: float | None = None
    blink_right: float | None = None
    smile: float | None = None
    jaw_open: float | None = None
    frame_id: int = 0
    taken_at: float = 0.0


def _average(points, ids, width, height):
    return (sum(points[i].x for i in ids) / len(ids) * width,
            sum(points[i].y for i in ids) / len(ids) * height)


class Detector:
    """Runs the face and hand models. Not thread safe: use one thread."""

    def __init__(self, face, hand, detect_width: int = DETECT_WIDTH, blendshapes: bool = True):
        self.face = face
        self.hand = hand
        self.detect_width = detect_width
        self.blendshapes = blendshapes
        self.faces_seen = 0
        self.hands_seen = 0
        self._last_ts = -1

    def _timestamp(self, now_ms: int) -> int:
        # MediaPipe needs strictly rising timestamps.
        self._last_ts = max(self._last_ts + 1, int(now_ms))
        return self._last_ts

    def detect(self, frame: np.ndarray, now_ms: int, frame_id: int = 0) -> Observation:
        import mediapipe as mp

        height, width = frame.shape[:2]
        small = frame
        if self.detect_width and width > self.detect_width:
            small = cv2.resize(frame, (self.detect_width, max(1, round(height * self.detect_width / width))),
                               interpolation=cv2.INTER_LINEAR)
        rgb = np.ascontiguousarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp = self._timestamp(now_ms)
        face_result = self.face.detect_for_video(image, timestamp)
        hand_result = self.hand.detect_for_video(image, timestamp)
        return self.build(face_result, hand_result, width, height, frame_id)

    def build(self, face_result, hand_result, width: int, height: int,
              frame_id: int = 0) -> Observation:
        self.faces_seen = len(face_result.face_landmarks)
        self.hands_seen = len(hand_result.hand_landmarks)
        obs = Observation(frame_id=frame_id, taken_at=time.perf_counter())
        if face_result.face_landmarks:
            points = face_result.face_landmarks[0]
            obs.face_found = True
            obs.nose = _average(points, NOSE_TIP_IDS, width, height)
            obs.forehead = _average(points, FOREHEAD_IDS, width, height)
            obs.chin = _average(points, CHIN_IDS, width, height)
            obs.mouth = _average(points, MOUTH_IDS, width, height)
            left, right = points[FACE_LEFT_ID], points[FACE_RIGHT_ID]
            obs.face_width = math.hypot((left.x - right.x) * width, (left.y - right.y) * height)
            blends = getattr(face_result, "face_blendshapes", None)
            if blends:
                scores = {c.category_name: c.score for c in blends[0]}
                obs.blink_left = scores.get("eyeBlinkLeft")
                obs.blink_right = scores.get("eyeBlinkRight")
                if "mouthSmileLeft" in scores and "mouthSmileRight" in scores:
                    obs.smile = (scores["mouthSmileLeft"] + scores["mouthSmileRight"]) / 2.0
                obs.jaw_open = scores.get("jawOpen")
        hands = [np.array([[p.x * width, p.y * height] for p in hand], dtype=np.float32)
                 for hand in hand_result.hand_landmarks]
        obs.hands = dedupe_hands(hands)
        return obs

    def close(self) -> None:
        try:
            self.face.close()
        finally:
            self.hand.close()


def create_detector(face_gestures: bool = True, detect_width: int = DETECT_WIDTH,
                    model_dir: Path | None = None) -> Detector:
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    model_dir = Path(model_dir) if model_dir else Path(__file__).resolve().parent / "models"
    face_model = model_dir / "face_landmarker.task"
    hand_model = model_dir / "hand_landmarker.task"
    for model in (face_model, hand_model):
        if not model.is_file():
            raise RuntimeError(f"Missing model file: {model}")
    face = vision.FaceLandmarker.create_from_options(
        vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(face_model)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            output_face_blendshapes=face_gestures,
            min_face_detection_confidence=0.4,
            min_face_presence_confidence=0.4,
            min_tracking_confidence=0.4,
        )
    )
    try:
        hand = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=str(hand_model)),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=2,
                min_hand_detection_confidence=0.4,
                min_hand_presence_confidence=0.4,
                min_tracking_confidence=0.4,
            )
        )
    except Exception:
        face.close()
        raise
    return Detector(face, hand, detect_width, blendshapes=face_gestures)


class DetectionWorker:
    """Runs a detector on a background thread.

    submit() hands over the newest picture (an older one that was never looked
    at is thrown away). latest() returns the newest finished Observation.
    With threaded=False everything runs inside submit(), which is what the
    tests use.
    """

    def __init__(self, detector, threaded: bool = True):
        self.detector = detector
        self.threaded = threaded
        self._cond = threading.Condition()
        self._pending = None
        self._latest: Observation | None = None
        self._stop = False
        self._error: Exception | None = None
        self._failures = 0
        self._started = time.perf_counter()
        self.detect_ms = 0.0
        self._thread = None
        if threaded:
            self._thread = threading.Thread(target=self._run, name="nirni-detect", daemon=True)
            self._thread.start()

    def _now_ms(self) -> int:
        return int((time.perf_counter() - self._started) * 1000)

    def submit(self, frame: np.ndarray, frame_id: int) -> None:
        if not self.threaded:
            self._detect(frame, frame_id)
            return
        with self._cond:
            self._pending = (frame, frame_id)
            self._cond.notify()

    def _detect(self, frame, frame_id) -> None:
        started = time.perf_counter()
        try:
            obs = self.detector.detect(frame, self._now_ms(), frame_id)
        except Exception as error:            # a bad frame must not kill the program
            self._failures += 1
            if self._failures >= 30:
                self._error = error
            return
        self._failures = 0
        self.detect_ms = 0.9 * self.detect_ms + 0.1 * (time.perf_counter() - started) * 1000.0
        self._latest = obs

    def _run(self) -> None:
        while True:
            with self._cond:
                while self._pending is None and not self._stop:
                    self._cond.wait(0.2)
                if self._stop:
                    return
                frame, frame_id = self._pending
                self._pending = None
            self._detect(frame, frame_id)

    def latest(self) -> Observation | None:
        if self._error is not None:
            error, self._error = self._error, None
            raise RuntimeError(f"Face and hand tracking keeps failing: {error}")
        return self._latest

    def close(self) -> None:
        with self._cond:
            self._stop = True
            self._cond.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self.detector.close()
