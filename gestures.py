"""Gesture logic for Nirni Livewebcam.

Copyright (c) 2026 Sidagam Nirni Naidu. All rights reserved.
Use of this file is governed by the LICENSE file in this folder.

Everything here is geometry and timing only, so it can be tested without a
camera. Positions are pixels.

  TouchDetector   a fingertip resting on the nose, forehead or chin
  WinkDetector    one eye closed, the other open, held a moment
  SmileDetector   a big smile held a moment (not just talking)
  GestureEngine   feeds the detectors from one observation of the trackers
  Dock            tool buttons you press by pointing at them and waiting
  WindowTracker   the rectangle between the index fingertips of two hands
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from air_canvas import INDEX_TIP_ID, is_pointing


# --------------------------------------------------------------------------
# Touch detection (nose, forehead, chin)
# --------------------------------------------------------------------------

class TouchDetector:
    """Decides when a fingertip has been resting on a point of the face."""

    def __init__(self, hold_seconds: float = 0.45, distance_ratio: float = 0.22,
                 target_memory: float = 0.7):
        self.hold_seconds = hold_seconds
        # A touch counts when the fingertip is closer to the target than
        # this fraction of the face width.
        self.distance_ratio = distance_ratio
        # A finger in front of the face can make the face tracker blink out
        # for a moment, so the last position is remembered briefly.
        self.target_memory = target_memory
        self.touch_since: float | None = None
        self.armed = True
        self.last_target: tuple[tuple[float, float], float, float] | None = None
        self.closest_ratio: float | None = None

    def disarm(self) -> None:
        """Require the finger to leave the target before the next trigger."""
        self.armed = False
        self.touch_since = None

    def update(self, target, face_width: float, fingertips, now: float) -> bool:
        """Feed one observation. Returns True once when a touch is confirmed."""
        if target is not None and face_width > 1.0:
            self.last_target = (target, face_width, now)
        elif self.last_target is not None and now - self.last_target[2] <= self.target_memory:
            target, face_width, _ = self.last_target
        else:
            target = None

        touching = False
        self.closest_ratio = None
        if target is not None and fingertips:
            closest = min(math.hypot(x - target[0], y - target[1]) for x, y in fingertips)
            self.closest_ratio = closest / face_width
            touching = self.closest_ratio <= self.distance_ratio

        if not touching:
            self.touch_since = None
            self.armed = True
            return False
        if not self.armed:
            return False
        if self.touch_since is None:
            self.touch_since = now
        if now - self.touch_since >= self.hold_seconds:
            self.disarm()
            return True
        return False


# --------------------------------------------------------------------------
# Face expressions
# --------------------------------------------------------------------------

class WinkDetector:
    """One eye closed while the other stays open, held for a moment.

    A normal blink closes both eyes, and an eye that is only half closing on
    its way into a blink does not last long enough, so neither counts.
    """

    def __init__(self, closed: float = 0.55, open_max: float = 0.35,
                 hold: float = 0.22, cooldown: float = 1.0):
        self.closed = closed
        self.open_max = open_max
        self.hold = hold
        self.cooldown = cooldown
        self.since: float | None = None
        self.armed = True
        self.last_fire = float("-inf")

    def reset(self) -> None:
        self.since = None
        self.armed = True

    def update(self, blink_left, blink_right, now: float) -> bool:
        if blink_left is None or blink_right is None:
            self.since = None
            return False
        winking = max(blink_left, blink_right) >= self.closed and \
            min(blink_left, blink_right) <= self.open_max
        if not winking:
            self.since = None
            self.armed = True
            return False
        if not self.armed:
            return False
        if self.since is None:
            self.since = now
        if now - self.since >= self.hold and now - self.last_fire >= self.cooldown:
            self.armed = False
            self.last_fire = now
            return True
        return False


class SmileDetector:
    """A wide smile held for a moment. Talking and a small smile do not count,
    and there is a long pause between two triggers."""

    def __init__(self, level: float = 0.55, hold: float = 0.45, cooldown: float = 6.0,
                 reset_level: float = 0.30, max_jaw_open: float = 0.45):
        self.level = level
        self.hold = hold
        self.cooldown = cooldown
        self.reset_level = reset_level
        self.max_jaw_open = max_jaw_open
        self.since: float | None = None
        self.armed = True
        self.last_fire = float("-inf")

    def reset(self) -> None:
        self.since = None

    def update(self, smile, jaw_open, now: float) -> bool:
        if smile is None:
            self.since = None
            return False
        if smile <= self.reset_level:
            self.armed = True
        smiling = smile >= self.level and (jaw_open is None or jaw_open <= self.max_jaw_open)
        if not smiling:
            self.since = None
            return False
        if not self.armed:
            return False
        if self.since is None:
            self.since = now
        if now - self.since >= self.hold and now - self.last_fire >= self.cooldown:
            self.armed = False
            self.last_fire = now
            self.since = None
            return True
        return False


@dataclass
class Events:
    """What the gesture engine decided for one observation."""
    nose: bool = False
    forehead: bool = False
    chin: bool = False
    wink: bool = False
    smile: bool = False


class GestureEngine:
    """Turns raw tracker observations into events."""

    def __init__(self, hold: float = 0.45, nose_distance: float = 0.22,
                 forehead_distance: float = 0.20, chin_distance: float = 0.20,
                 face_gestures: bool = True, smile_level: float = 0.55):
        self.nose = TouchDetector(hold, nose_distance)
        self.forehead = TouchDetector(hold, forehead_distance)
        self.chin = TouchDetector(hold, chin_distance)
        self.wink = WinkDetector()
        self.smile = SmileDetector(level=smile_level)
        self.face_gestures = face_gestures

    def disarm_touches(self) -> None:
        for detector in (self.nose, self.forehead, self.chin):
            detector.disarm()

    def process(self, obs, now: float, suppress: bool = False) -> Events:
        """While suppress is True no touch can trigger, for example while you are
        drawing or carrying a shape."""
        fingertips = [(float(h[INDEX_TIP_ID][0]), float(h[INDEX_TIP_ID][1])) for h in obs.hands]
        hits = {}
        for name, detector, target in (("nose", self.nose, obs.nose),
                                       ("forehead", self.forehead, obs.forehead),
                                       ("chin", self.chin, obs.chin)):
            if detector.update(target, obs.face_width, fingertips, now):
                hits[name] = detector.closest_ratio if detector.closest_ratio is not None else 9.0
        events = Events()
        if suppress:
            self.disarm_touches()
        elif hits:
            setattr(events, min(hits, key=hits.get), True)

        if self.face_gestures and obs.face_found and not suppress:
            events.wink = self.wink.update(obs.blink_left, obs.blink_right, now)
            events.smile = self.smile.update(obs.smile, obs.jaw_open, now)
        else:
            self.wink.reset()
            self.smile.reset()
        return events


# --------------------------------------------------------------------------
# The tool dock: point at a button and wait
# --------------------------------------------------------------------------

@dataclass
class Button:
    key: str
    label: str
    rect: tuple[int, int, int, int]            # x0, y0, x1, y1
    color: tuple[int, int, int] | None = None   # swatch colour, if any
    side: str = "R"


class Dock:
    """Two columns of buttons along the picture edges. A button is pressed by
    pointing at it with the index finger and holding still for a moment.

    The dock is drawn in the preview window only, so people on your call
    never see it.
    """

    def __init__(self, dwell: float = 0.7, drop_grace: float = 0.25):
        self.dwell = dwell
        self.drop_grace = drop_grace
        self.buttons: list[Button] = []
        self.hover: str | None = None
        self.progress = 0.0
        self._key: str | None = None
        self._since = 0.0
        self._lost_at: float | None = None
        self._fired: str | None = None

    def layout(self, width: int, height: int, left_items, right_items) -> None:
        """Items are (key, label, colour or None). Left column on the left
        edge, right column on the right edge."""
        self.buttons = []
        for side, items in (("L", left_items), ("R", right_items)):
            if not items:
                continue
            size = int(max(26, min(54, (height - 30) // max(len(items), 1))))
            gap = 4
            button_width = int(max(48, min(72, width // 14)))
            total = len(items) * size + (len(items) - 1) * gap
            top = max(8, (height - total) // 2)
            x0 = 8 if side == "L" else width - 8 - button_width
            for i, (key, label, color) in enumerate(items):
                y0 = top + i * (size + gap)
                self.buttons.append(Button(key, label, (x0, y0, x0 + button_width, y0 + size),
                                           color, side))
        self.reset()

    def reset(self) -> None:
        self.hover = None
        self.progress = 0.0
        self._key = None
        self._lost_at = None
        self._fired = None

    def button_at(self, point, pad: int = 6) -> Button | None:
        x, y = float(point[0]), float(point[1])
        for button in self.buttons:
            x0, y0, x1, y1 = button.rect
            if x0 - pad <= x <= x1 + pad and y0 - pad <= y <= y1 + pad:
                return button
        return None

    def update(self, hands, now: float, active: bool = True) -> str | None:
        """Returns the key of a button once, when its dwell time is complete."""
        if not active or not self.buttons:
            self.reset()
            return None
        target = None
        for hand in hands:
            if is_pointing(hand):
                button = self.button_at(hand[INDEX_TIP_ID])
                if button is not None:
                    target = button
                    break
        if target is None:
            if self._key is not None:
                if self._lost_at is None:
                    self._lost_at = now
                elif now - self._lost_at > self.drop_grace:
                    self.reset()
            return None

        self._lost_at = None
        if target.key != self._key:
            self._key = target.key
            self._since = now
            if self._fired != target.key:
                self._fired = None
        self.hover = target.key
        if self._fired == target.key:         # already pressed, wait until the finger leaves
            self.progress = 0.0
            return None
        self.progress = min(1.0, (now - self._since) / self.dwell)
        if self.progress >= 1.0:
            self._fired = target.key
            self.progress = 0.0
            return target.key
        return None


# --------------------------------------------------------------------------
# The effect window: a rectangle between two index fingertips
# --------------------------------------------------------------------------

class WindowTracker:
    """The rectangle whose opposite corners are the index fingertips of two
    pointing hands. Smoothed, and it survives a short tracking drop-out."""

    def __init__(self, min_size: int = 48, smoothing: float = 0.6, grace: float = 0.25):
        self.min_size = min_size
        self.smoothing = smoothing
        self.grace = grace
        self.rect: tuple[int, int, int, int] | None = None
        self._float = None
        self._lost_at: float | None = None

    @property
    def active(self) -> bool:
        return self.rect is not None

    def reset(self) -> None:
        self.rect = None
        self._float = None
        self._lost_at = None

    def update(self, hands, now: float, enabled: bool = True, bounds=None):
        import numpy as np

        found = None
        if enabled:
            tips = [h[INDEX_TIP_ID] for h in hands if is_pointing(h)]
            if len(tips) >= 2:
                a, b = tips[0], tips[1]
                x0, x1 = sorted((float(a[0]), float(b[0])))
                y0, y1 = sorted((float(a[1]), float(b[1])))
                if x1 - x0 >= self.min_size and y1 - y0 >= self.min_size:
                    found = np.array([x0, y0, x1, y1], dtype=np.float32)
        if found is not None:
            self._lost_at = None
            if self._float is None:
                self._float = found
            else:
                self._float = self._float + self.smoothing * (found - self._float)
        elif self._float is not None:
            if self._lost_at is None:
                self._lost_at = now
            if not enabled or now - self._lost_at > self.grace:
                self.reset()
                return None
        if self._float is None:
            self.rect = None
            return None
        x0, y0, x1, y1 = (int(round(v)) for v in self._float)
        if bounds is not None:
            width, height = bounds
            x0, x1 = max(0, x0), min(width, x1)
            y0, y1 = max(0, y0), min(height, y1)
        self.rect = (x0, y0, x1, y1) if x1 - x0 >= 8 and y1 - y0 >= 8 else None
        return self.rect
