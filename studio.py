"""The brain of Nirni Livewebcam: modes, gestures, effects, in one place.

Copyright (c) 2026 Sidagam Nirni Naidu. All rights reserved.
Use of this file is governed by the LICENSE file in this folder.

Studio takes a picture and what the trackers saw in it, and returns the clean
picture that goes to the virtual camera. It never touches a camera or a
window, which is why all of it can be tested.

Order in which a picture is built:
  live picture or vanish effect -> ghost trail / clones -> shapes and the
  stroke being drawn -> effect window -> confetti and smoke
"""

from __future__ import annotations

import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from air_canvas import COLORS, RAINBOW_INDEX, AirPen, ShapeBoard, is_pointing, recognize_shape, INDEX_TIP_ID
from effects import (WINDOW_KINDS, BackgroundModel, GhostEngine, ParticleSystem,
                     PersonMasker, SegmenterLoader, Vanisher, apply_window)
from gestures import Dock, GestureEngine, WindowTracker
from media import Recorder, save_snapshot
from tracking import Observation

CALIBRATION_SECONDS = 4.0     # at least this long before the background is saved
CALIBRATION_CLEAR = 1.0       # the picture must have been empty and still this long
CALIBRATION_TIMEOUT = 30.0    # then it is saved anyway, with a warning
MOTION_LIMIT = 6.0            # mean grey-level change between frames that counts as "moving"
VANISH_SECONDS = 3.0
APPEAR_SECONDS = 1.8
TOAST_SECONDS = 2.0
STALE_OBSERVATION = 0.5       # s, tracker results older than this are treated as "nothing seen"
SNAPSHOT_COUNTDOWN = 3.0

MODES = ("normal", "draw", "window")
WINDOW_LABELS = {"blur": "BLUR", "pixelate": "PIX", "magnify": "ZOOM"}


class Studio:
    def __init__(self, width: int, height: int, *, hold: float = 0.45,
                 nose_distance: float = 0.22, forehead_distance: float = 0.20,
                 chin_distance: float = 0.20, pen_join: float = 0.55,
                 face_gestures: bool = True, smile_level: float = 0.55,
                 style: str = "dissolve", output_dir: Path | str = "captures",
                 loader: SegmenterLoader | None = None, fps: float = 30.0):
        self.width, self.height = width, height
        self.phase = "calibrating"          # calibrating, ready, vanishing, vanished, appearing
        self.phase_started = 0.0
        self.mode = "normal"                # normal, draw, window
        self.tool = "pen"                   # pen, eraser (inside draw mode)
        self.window_kind = "blur"
        self.vanish_style = style if style in Vanisher.STYLES else "dissolve"
        self.help = False
        self.toast = ""
        self.toast_until = 0.0
        self.flash_until = 0.0
        self.snapshot_at: float | None = None
        self.output_dir = Path(output_dir)

        self.background = BackgroundModel()
        self.loader = loader
        self.masker = PersonMasker(self.background, loader)
        self.gestures = GestureEngine(hold, nose_distance, forehead_distance, chin_distance,
                                      face_gestures, smile_level)
        self.pen = AirPen(join_ratio=pen_join)
        self.board = ShapeBoard(width, height)
        self.dock = Dock()
        self.window = WindowTracker()
        self.ghost = GhostEngine()
        self.recorder = Recorder(self.output_dir, fps)
        self.vanisher = Vanisher(width, height)
        self.particles = ParticleSystem(width, height)

        self._samples: deque[np.ndarray] = deque(maxlen=14)
        self._cal_prev: np.ndarray | None = None
        self._cal_ok_since: float | None = None
        self._cal_force = False
        self.cal_status = ""
        self._seen_id = -1
        self._last_obs_at = 0.0
        self._latest_obs = Observation()
        self._mouth: tuple[float, float] | None = None
        self._last_frame: np.ndarray | None = None
        self._last_adapt = 0.0
        self._eraser_hits = 0
        self.last_mask: np.ndarray | None = None
        self._refresh_dock()

    # -- small helpers -------------------------------------------------------

    def say(self, text: str, now: float) -> None:
        self.toast = text
        self.toast_until = now + TOAST_SECONDS

    @staticmethod
    def _ms(now: float) -> int:
        return int(now * 1000)

    @property
    def face_gestures(self) -> bool:
        return self.gestures.face_gestures

    # -- size changes --------------------------------------------------------

    def _ensure_size(self, frame: np.ndarray) -> None:
        height, width = frame.shape[:2]
        if (width, height) == (self.width, self.height):
            return
        self.width, self.height = width, height
        self.board.rescale(width, height)
        self.vanisher = Vanisher(width, height)
        self.particles = ParticleSystem(width, height)
        self.ghost.clear()
        self.pen.reset()
        self.window.reset()
        if self.background.ready:
            self.background.set(cv2.resize(self.background.base, (width, height)))
        self._refresh_dock()

    # -- dock ----------------------------------------------------------------

    def _refresh_dock(self) -> None:
        if self.mode == "normal":
            self.dock.layout(self.width, self.height, [], [])
            return
        left = [("blur", "BLUR", None), ("pixelate", "PIX", None), ("magnify", "ZOOM", None),
                ("ghost", "GHOST", None), ("clone", "CLONE", None),
                ("snap", "SNAP", None), ("rec", "REC", None)]
        right = []
        if self.mode == "draw":
            right = [(f"c{i}", name.upper()[:5], color) for i, (name, color) in enumerate(COLORS)]
            right += [("rainbow", "RAIN", None), ("eraser", "ERASE", None),
                      ("undo", "UNDO", None), ("redo", "REDO", None)]
        self.dock.layout(self.width, self.height, left, right)

    def _dock_action(self, key: str, now: float) -> None:
        if key.startswith("c") and key[1:].isdigit():
            self.board.set_color(int(key[1:]))
            self.tool = "pen"
            self.say(f"Colour: {self.board.color_name()}", now)
        elif key == "rainbow":
            self.board.set_color(RAINBOW_INDEX)
            self.tool = "pen"
            self.say("Colour: Rainbow", now)
        elif key == "eraser":
            self.toggle_eraser(now)
        elif key == "undo":
            self.undo(now)
        elif key == "redo":
            self.redo(now)
        elif key in WINDOW_KINDS:
            self.window_kind = key
            self.set_mode("window", now)
            self.say(f"Window: {WINDOW_LABELS[key]}", now)
        elif key == "ghost":
            self.cycle_ghost(now)
        elif key == "clone":
            self.spawn_clone(now)
        elif key == "snap":
            self.snapshot_at = now + SNAPSHOT_COUNTDOWN
            self.say("Photo in 3 seconds", now)
        elif key == "rec":
            self.toggle_recording(now)

    # -- modes ---------------------------------------------------------------

    def set_mode(self, mode: str, now: float) -> None:
        if mode not in MODES:
            raise ValueError(mode)
        if mode == self.mode:
            return
        self.mode = mode
        self.pen.reset()
        self.window.reset()
        self.dock.reset()
        self._refresh_dock()
        text = {"normal": "Tools OFF", "draw": "Draw mode ON", "window": f"Window: {WINDOW_LABELS[self.window_kind]}"}
        self.say(text[mode], now)

    def toggle_draw(self, now: float) -> None:
        self.set_mode("normal" if self.mode == "draw" else "draw", now)

    def toggle_window(self, now: float) -> None:
        self.set_mode("normal" if self.mode == "window" else "window", now)

    def cycle_window(self, now: float) -> None:
        """off -> blur -> pixelate -> magnify -> off"""
        if self.mode != "window":
            self.window_kind = WINDOW_KINDS[0]
            self.set_mode("window", now)
            return
        index = WINDOW_KINDS.index(self.window_kind) + 1
        if index >= len(WINDOW_KINDS):
            self.set_mode("normal", now)
        else:
            self.window_kind = WINDOW_KINDS[index]
            self.say(f"Window: {WINDOW_LABELS[self.window_kind]}", now)

    def toggle_eraser(self, now: float) -> None:
        if self.mode != "draw":
            self.set_mode("draw", now)
        self.tool = "pen" if self.tool == "eraser" else "eraser"
        self.pen.reset()
        self.say("Eraser: point at a shape" if self.tool == "eraser" else "Pen", now)

    # -- history -------------------------------------------------------------

    def undo(self, now: float) -> None:
        self.say("Undo" if self.board.undo() else "Nothing to undo", now)

    def redo(self, now: float) -> None:
        self.say("Redo" if self.board.redo() else "Nothing to redo", now)

    # -- ghost, clones, confetti ----------------------------------------------

    def cycle_ghost(self, now: float) -> None:
        mode = self.ghost.cycle()
        self.say({"off": "Ghost OFF", "trail": "Ghost trail", "clones": "Shadow clones"}[mode], now)

    def spawn_clone(self, now: float) -> None:
        frame = self._last_frame
        if frame is None or self.phase == "calibrating":
            self.say("Save the background first", now)
            return
        mask = self.masker.compute(frame, self._ms(now))
        if mask is None or not self.ghost.add_clone(frame, mask, now):
            self.say("Step into the picture first", now)
            return
        if self.ghost.mode != "clones":
            self.ghost.mode = "clones"
        box = self.ghost.clones[-1]
        h, w = box.alpha.shape
        self.particles.puff(box.origin[0] + w / 2, box.origin[1] + h / 2, spread=w * 0.6)
        self.say("Shadow clone!", now)

    def burst(self) -> None:
        x, y = self._mouth if self._mouth else (None, None)
        self.particles.confetti(x, y)

    # -- photos and video ------------------------------------------------------

    def take_snapshot(self, display: np.ndarray, now: float) -> None:
        try:
            path = save_snapshot(display, self.output_dir)
            self.flash_until = now + 0.25
            self.say(f"Saved {path.name}", now)
            print(f"[Nirni Livewebcam] Photo saved: {path}")
        except (OSError, RuntimeError) as error:
            self.say("Could not save the photo", now)
            print(f"[Nirni Livewebcam] Photo failed: {error}")

    def toggle_recording(self, now: float) -> None:
        if self.recorder.recording:
            path, frames = self.recorder.stop()
            self.say(f"Saved {path.name}" if path else "Recording stopped", now)
            print(f"[Nirni Livewebcam] Video saved: {path} ({frames} frames, no sound)")
            return
        try:
            path = self.recorder.start((self.width, self.height), now)
        except (OSError, RuntimeError) as error:
            self.say("Could not start recording", now)
            print(f"[Nirni Livewebcam] Recording failed: {error}")
            return
        self.say("Recording...", now)
        print(f"[Nirni Livewebcam] Recording to {path}")

    def close(self) -> None:
        if self.recorder.recording:
            self.recorder.stop()

    # -- vanish ----------------------------------------------------------------

    def _vanish_progress(self, now: float) -> float:
        age = now - self.phase_started
        if self.phase == "vanishing":
            return min(1.0, age / VANISH_SECONDS)
        if self.phase == "appearing":
            return max(0.0, 1.0 - age / APPEAR_SECONDS)
        return 1.0 if self.phase == "vanished" else 0.0

    def begin_vanish(self, now: float) -> None:
        if self.phase == "ready":
            self.phase, self.phase_started = "vanishing", now
        elif self.phase == "appearing":      # turn round from where the fade-in had got to
            progress = self._vanish_progress(now)
            self.phase, self.phase_started = "vanishing", now - progress * VANISH_SECONDS
        else:
            return
        self.gestures.nose.disarm()

    def begin_appear(self, now: float) -> None:
        if self.phase not in ("vanishing", "vanished"):
            return
        progress = self._vanish_progress(now)
        self.phase = "appearing"
        self.phase_started = now - (1.0 - progress) * APPEAR_SECONDS
        self.gestures.nose.disarm()

    def next_style(self, now: float) -> None:
        styles = Vanisher.STYLES
        self.vanish_style = styles[(styles.index(self.vanish_style) + 1) % len(styles)]
        self.say(f"Vanish style: {self.vanish_style}", now)

    def start_calibration(self, now: float) -> None:
        self.phase, self.phase_started = "calibrating", now
        self._samples.clear()
        self._cal_prev = None
        self._cal_ok_since = None
        self._cal_force = False
        self.pen.reset()
        self.window.reset()
        self.board.flush()

    # -- keys ------------------------------------------------------------------

    def on_key(self, ch: str, now: float) -> str | None:
        """Handle one key (a lower-case letter, digit or symbol). Returns
        'quit' or 'fullscreen' for the caller, otherwise None."""
        if ch in ("q", "\x1b"):
            return "quit"
        if ch == "f":
            return "fullscreen"
        if ch == " ":
            self.start_calibration(now)
            self.say("Step out of the frame", now)
        elif ch == "\r" or ch == "\n":
            self._cal_force = self.phase == "calibrating"
        elif ch == "v" and self.phase == "ready":
            self.begin_vanish(now)
        elif ch == "r":
            self.begin_appear(now)
        elif ch == "t":
            self.next_style(now)
        elif ch == "d":
            self.toggle_draw(now)
        elif ch == "w":
            self.cycle_window(now)
        elif ch == "e":
            self.toggle_eraser(now)
        elif ch == "g":
            self.cycle_ghost(now)
        elif ch == "n":
            self.spawn_clone(now)
        elif ch == "z":
            self.undo(now)
        elif ch == "y":
            self.redo(now)
        elif ch == "c":
            self.say("All shapes cleared (Z brings them back)" if self.board.clear() else "Nothing to clear", now)
        elif ch in "1234567":
            self.board.set_color(int(ch) - 1)
            self.tool = "pen" if self.tool == "eraser" and self.mode == "draw" else self.tool
            self.say(f"Colour: {self.board.color_name()}", now)
        elif ch == "8":
            self.board.set_color(RAINBOW_INDEX)
            self.say("Colour: Rainbow", now)
        elif ch == "0":
            self.board.set_color(None)
            self.say("Colour: changes per shape", now)
        elif ch == "[":
            self.say(f"Pen size {self.board.change_thickness(-1)}", now)
        elif ch == "]":
            self.say(f"Pen size {self.board.change_thickness(+1)}", now)
        elif ch == "p":
            self.snapshot_at = now            # taken at the end of the next picture
        elif ch == "m":
            self.toggle_recording(now)
        elif ch == "b":
            self.burst()
        elif ch == "s":
            self.gestures.face_gestures = not self.gestures.face_gestures
            self.say("Wink / smile ON" if self.gestures.face_gestures else "Wink / smile OFF", now)
        elif ch == "h":
            self.help = not self.help
        return None

    # -- calibration -----------------------------------------------------------

    def _calibrate(self, frame: np.ndarray, obs: Observation, now: float) -> None:
        present = obs.face_found or bool(obs.hands)
        grey = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (64, 36)).astype(np.float32)
        moving = self._cal_prev is not None and float(np.abs(grey - self._cal_prev).mean()) > MOTION_LIMIT
        self._cal_prev = grey
        if present or moving:
            self._cal_ok_since = None
            self._samples.clear()
            self.cal_status = "Step out of the frame" if present else "Hold still"
        else:
            if self._cal_ok_since is None:
                self._cal_ok_since = now
            if now - self._cal_ok_since >= 0.4:
                self._samples.append(frame)
            self.cal_status = "Saving the background"
        elapsed = now - self.phase_started
        clear_long_enough = self._cal_ok_since is not None and now - self._cal_ok_since >= CALIBRATION_CLEAR
        if (elapsed >= CALIBRATION_SECONDS and clear_long_enough and len(self._samples) >= 4) \
                or self._cal_force or elapsed >= CALIBRATION_TIMEOUT:
            self._finish_calibration(frame, now, clean=not (self._cal_force or elapsed >= CALIBRATION_TIMEOUT))

    def _finish_calibration(self, frame: np.ndarray, now: float, clean: bool) -> None:
        if self._samples:
            self.background.set(np.median(np.stack(list(self._samples)), axis=0).astype(np.uint8))
        else:
            self.background.set(frame)
        self._samples.clear()
        self._cal_force = False
        self.phase = "ready"
        self.gestures.disarm_touches()
        self.say("Background saved" if clean else "Saved - press SPACE if you were in the picture", now)
        print("[Nirni Livewebcam] Background saved. You can step back in." if clean else
              "[Nirni Livewebcam] Background saved without a clean empty picture.")

    # -- what the trackers saw ---------------------------------------------------

    def _handle_observation(self, obs: Observation, now: float) -> None:
        hands = obs.hands
        bounds = (self.width, self.height)
        self._mouth = obs.mouth or self._mouth
        self.window.update(hands, now, enabled=self.mode == "window", bounds=bounds)
        key = self.dock.update(hands, now, active=self.mode != "normal" and not self.window.active
                               and not self.pen.down)
        if key:
            self._dock_action(key, now)

        suppress = (self.pen.down or self.board.holding or self.dock.hover is not None
                    or self.window.active)
        events = self.gestures.process(obs, now, suppress=suppress)
        if events.forehead:
            self.toggle_draw(now)
        if events.chin:
            self.toggle_window(now)
        if events.nose:
            if self.phase in ("ready", "appearing"):
                self.begin_vanish(now)
            elif self.phase in ("vanishing", "vanished"):
                self.begin_appear(now)
        if events.wink:
            self.undo(now)
        if events.smile:
            self.burst()
            self.say("Smile!", now)

        pen_on = self.mode == "draw" and self.tool == "pen"
        finished = self.pen.update(hands, now, enabled=pen_on)
        if finished is not None:
            shape = recognize_shape(finished, self.board.next_color(), self.board.thickness,
                                    self.board.rainbow)
            if shape is not None:
                self.board.add(shape)
        self.board.update_hands(hands, now, allow_grab=not self.pen.down)

        self._eraser_hits = 0
        if self.mode == "draw" and self.tool == "eraser":
            tips = [h[INDEX_TIP_ID] for h in hands
                    if is_pointing(h) and self.dock.button_at(h[INDEX_TIP_ID]) is None]
            self._eraser_hits = self.board.erase_at(tips, self.eraser_radius, now)

    @property
    def eraser_radius(self) -> float:
        return 18.0 + 4.0 * self.board.thickness

    # -- one picture -------------------------------------------------------------

    def process(self, frame: np.ndarray, obs: Observation | None, now: float) -> np.ndarray:
        """Build the clean output picture for one camera frame."""
        self._ensure_size(frame)
        self._last_frame = frame
        fresh = None
        if obs is not None and obs.frame_id != self._seen_id:
            self._seen_id = obs.frame_id
            self._last_obs_at = now
            fresh = self._latest_obs = obs
        elif now - self._last_obs_at > STALE_OBSERVATION:
            fresh = self._latest_obs = Observation()      # tracker went quiet: nothing is seen

        if self.phase == "calibrating":
            self._calibrate(frame, self._latest_obs, now)
        elif fresh is not None:
            self._handle_observation(fresh, now)
        self.board.tick(now)

        if self.phase == "vanishing" and self._vanish_progress(now) >= 1.0:
            self.phase = "vanished"
        elif self.phase == "appearing" and self._vanish_progress(now) <= 0.0:
            self.phase = "ready"

        in_transition = self.phase in ("vanishing", "appearing")
        mask = None
        if (in_transition or self.ghost.needs_mask or (self.ghost.clones and self.phase != "calibrating")) \
                and self.masker.available:
            mask = self.masker.compute(frame, self._ms(now))
        self.last_mask = mask

        if self.background.ready and self.phase != "calibrating" and now - self._last_adapt > 0.35:
            self._last_adapt = now
            self.background.adapt(frame, mask if in_transition else None)

        if self.phase in ("ready", "calibrating"):
            canvas = frame.copy()
        elif self.phase == "vanished":
            canvas = self.background.current().copy()
        else:
            canvas = self.vanisher.render(self.vanish_style, frame, self.background.current(),
                                          mask, self._vanish_progress(now), now)

        self.ghost.feed(frame, mask, now)
        self.ghost.render(canvas, now, live_mask=mask if self.phase == "ready" else None)

        stroke = self.pen.points if self.pen.down else None
        self.board.draw(canvas, stroke, self.board.next_color(), self.board.rainbow)

        if self.mode == "window" and self.window.rect is not None:
            apply_window(canvas, self.window.rect, self.window_kind)

        self.particles.update(now)
        self.particles.render(canvas)

        if self.snapshot_at is not None and now >= self.snapshot_at:
            self.snapshot_at = None
            self.take_snapshot(canvas, now)
        self.recorder.write(canvas, now)
        return canvas
