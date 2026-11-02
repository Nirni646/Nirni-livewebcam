"""Preview-window overlay for Nirni Livewebcam.

Copyright (c) 2026 Sidagam Nirni Naidu. All rights reserved.
Use of this file is governed by the LICENSE file in this folder.

Everything drawn here is for you only. The virtual camera and the saved
photos and videos get the clean picture, without any of this.
"""

from __future__ import annotations

import cv2
import numpy as np

from air_canvas import INDEX_TIP_ID, is_pointing

HELP_LINES = (
    "TOUCH   nose = vanish / return    forehead = draw tools    chin = effect window",
    "FACE    wink = undo    big smile = confetti",
    "DRAW    join both index tips to draw, pull apart to finish a shape",
    "MOVE    fist over a shape carries it    two fists resize + rotate it",
    "WINDOW  point with both index fingers: the box between them gets the effect",
    "TOOLS   point at a button on the screen edge and wait",
    "KEYS    SPACE background   V vanish   R return   T vanish style",
    "        D draw   E eraser   W window   G ghost   N new clone",
    "        Z undo   Y redo   C clear   1-7 colour   8 rainbow   0 auto   [ ] size",
    "        P photo   M record   B confetti   S wink/smile on-off",
    "        H help   F fullscreen   Q quit",
)


def put_text(image: np.ndarray, text: str, origin: tuple[int, int], scale: float = 0.6,
             color=(255, 255, 255)) -> None:
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def _button_active(studio, key: str) -> bool:
    board = studio.board
    if key.startswith("c") and key[1:].isdigit():
        return board.color_index == int(key[1:]) and studio.tool == "pen"
    return {
        "rainbow": board.rainbow and studio.tool == "pen",
        "eraser": studio.tool == "eraser",
        "ghost": studio.ghost.mode != "off",
        "rec": studio.recorder.recording,
        "undo": board.can_undo,
        "redo": board.can_redo,
    }.get(key, studio.mode == "window" and studio.window_kind == key)


def _draw_dock(out: np.ndarray, studio) -> None:
    for button in studio.dock.buttons:
        x0, y0, x1, y1 = button.rect
        active = _button_active(studio, button.key)
        hover = studio.dock.hover == button.key
        cv2.rectangle(out, (x0, y0), (x1, y1), (35, 35, 35), -1)
        if button.color is not None:
            cv2.rectangle(out, (x0 + 4, y0 + 4), (x0 + 16, y1 - 4), button.color, -1)
        elif button.key == "rainbow":
            for i in range(12):
                hsv = np.uint8([[[int(i * 15), 235, 255]]])
                bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
                top = y0 + 4 + i * (y1 - y0 - 8) // 12
                cv2.rectangle(out, (x0 + 4, top), (x0 + 16, top + (y1 - y0) // 12 + 1),
                              tuple(int(c) for c in bgr), -1)
        edge = (80, 230, 255) if hover else (120, 255, 140) if active else (110, 110, 110)
        cv2.rectangle(out, (x0, y0), (x1, y1), edge, 2 if (hover or active) else 1)
        label_x = x0 + (22 if button.color is not None or button.key == "rainbow" else 6)
        dim = not active and button.key in ("undo", "redo")
        put_text(out, button.label, (label_x, y0 + (y1 - y0) // 2 + 5), 0.4,
                 (140, 140, 140) if dim else (255, 255, 255))
        if hover and studio.dock.progress > 0:
            bar = x0 + int((x1 - x0) * studio.dock.progress)
            cv2.rectangle(out, (x0, y1 - 5), (bar, y1), (80, 230, 255), -1)


def draw_hud(studio, display: np.ndarray, now: float, fps: float, show_fps: bool = True) -> np.ndarray:
    """Return a copy of the picture with hints for you. Never sent anywhere."""
    out = display.copy()
    height, width = out.shape[:2]
    obs = studio._latest_obs

    if studio.phase == "calibrating":
        put_text(out, studio.cal_status or "Step out of the frame", (16, 34), 0.8, (80, 230, 255))
        put_text(out, "Saving a picture of the empty background. ENTER saves right now.",
                 (16, 62), 0.5)

    if studio.mode != "normal":
        label = "DRAW" if studio.mode == "draw" else f"WINDOW {studio.window_kind.upper()}"
        if studio.mode == "draw":
            label += f"  {'ERASER' if studio.tool == 'eraser' else 'PEN'}  {studio.board.color_name()}  size {studio.board.thickness}"
        put_text(out, label, (16, 30 if studio.phase != "calibrating" else 92), 0.65, (120, 255, 140))
        _draw_dock(out, studio)
        for hand in obs.hands:
            tip = hand[INDEX_TIP_ID]
            pointing = is_pointing(hand)
            cv2.circle(out, (int(tip[0]), int(tip[1])), 9 if pointing else 5,
                       (80, 230, 255) if pointing else (120, 255, 140), 2, cv2.LINE_AA)
        if studio.pen.cursor is not None:
            color = (80, 230, 255) if studio.pen.down else (255, 255, 255)
            cv2.circle(out, (int(studio.pen.cursor[0]), int(studio.pen.cursor[1])), 11, color, 2, cv2.LINE_AA)
        if studio.tool == "eraser" and studio.mode == "draw":
            for hand in obs.hands:
                if is_pointing(hand):
                    tip = hand[INDEX_TIP_ID]
                    cv2.circle(out, (int(tip[0]), int(tip[1])), int(studio.eraser_radius),
                               (80, 90, 255), 2, cv2.LINE_AA)

    line = 56
    if studio.board.manipulating:
        put_text(out, "two hands: resize + rotate", (16, line), 0.55, (255, 200, 60))
    elif studio.board.holding:
        put_text(out, "holding a shape", (16, line), 0.55, (255, 200, 60))
    if studio.ghost.mode != "off":
        put_text(out, f"ghost: {studio.ghost.mode}", (16, line + 24), 0.5, (255, 170, 80))
    if not studio.face_gestures:
        put_text(out, "wink / smile off", (16, height - 40), 0.45, (150, 150, 150))
    if studio.recorder.recording:
        seconds = int(studio.recorder.elapsed(now))
        cv2.circle(out, (width - 150, 24), 8, (60, 60, 255), -1, cv2.LINE_AA)
        put_text(out, f"REC {seconds // 60:02d}:{seconds % 60:02d}", (width - 134, 30), 0.6, (60, 60, 255))
    if show_fps:
        put_text(out, f"{fps:.0f} FPS", (width - 90, height - 14), 0.5, (200, 200, 200))
    if studio.snapshot_at is not None and studio.snapshot_at > now:
        text = str(int(np.ceil(studio.snapshot_at - now)))
        size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 4.0, 6)[0]
        put_text(out, text, ((width - size[0]) // 2, (height + size[1]) // 2), 4.0, (255, 255, 255))
    if now < studio.flash_until:
        alpha = float(np.clip((studio.flash_until - now) / 0.25, 0, 1)) * 0.7
        out = cv2.addWeighted(out, 1.0 - alpha, np.full_like(out, 255), alpha, 0)
    if now < studio.toast_until and studio.toast:
        size = cv2.getTextSize(studio.toast, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
        put_text(out, studio.toast, ((width - size[0]) // 2, 70), 0.9, (80, 230, 255))

    if studio.help:
        shade = out.copy()
        cv2.rectangle(shade, (30, 90), (width - 30, 90 + 28 * len(HELP_LINES) + 20), (0, 0, 0), -1)
        out = cv2.addWeighted(shade, 0.72, out, 0.28, 0)
        for i, text in enumerate(HELP_LINES):
            put_text(out, text, (46, 125 + 28 * i), 0.55)
    elif studio.mode == "normal" and studio.phase != "calibrating":
        put_text(out, "H = help", (16, height - 14), 0.45, (200, 200, 200))
    return out
