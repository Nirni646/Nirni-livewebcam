"""Air canvas for Nirni Livewebcam.

Copyright (c) 2026 Sidagam Nirni Naidu. All rights reserved.
Use of this file is governed by the LICENSE file in this folder.

Pieces that live here, none of them need a camera to be tested:

  hand maths     palm size, fist / open hand, pointing finger, duplicate hands
  AirPen         joins the index fingertips of both hands into one pen
  recognize_shape  turns the drawn path into a clean line, circle, triangle,
                 square / rectangle, or keeps it as a smooth freehand curve
  ShapeBoard     keeps every shape, colours, eraser, undo / redo history,
                 lets a closed hand catch a shape and carry it, lets two closed
                 hands resize and rotate it, and paints everything on the video

All positions are pixels in the (already mirrored) webcam frame.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

# Hand landmark numbers (MediaPipe hand model).
WRIST_ID = 0
MIDDLE_MCP_ID = 9
INDEX_TIP_ID = 8
FINGERTIP_IDS = (8, 12, 16, 20)
FINGER_PIP_IDS = (6, 10, 14, 18)
PALM_IDS = (0, 5, 9, 13, 17)

# finger_reach() below this means a closed hand, above it an open one.
# Two values so a hand does not flicker between the states.
FIST_CLOSE = 1.35
FIST_OPEN = 1.55

# A pointing hand: the index finger is stretched, at least two of the other
# three are curled. Ratios are tip-to-wrist divided by knuckle-to-wrist.
POINT_INDEX_MIN = 1.05
POINT_CURL_MAX = 1.0

MIN_STROKE_POINTS = 8
MIN_SHAPE_SIZE = 45.0       # px, smaller scribbles are ignored
MAX_SHAPES = 30
HISTORY_LIMIT = 60

GRAB_MARGIN = 45.0          # px around a shape where a closed hand can catch it
REACH_WINDOW = 0.8          # s, an open hand must have been near the shape this recently
FOLLOW_DISTANCE = 260.0     # px, how far the hand may jump between two checks
RELEASE_GRACE = 0.3         # s, tracking drop-outs shorter than this keep the grip
MIN_SCALE = 0.3             # two-hand resize limits
MAX_SCALE = 4.0
ERASE_COMMIT_DELAY = 0.5    # s, one sweep of the eraser is one undo step

# Named colours (BGR). Index 7 is rainbow, None means "change colour per shape".
COLORS = (
    ("Sky", (255, 200, 60)),
    ("Mint", (120, 255, 140)),
    ("Orange", (90, 170, 255)),
    ("Violet", (255, 120, 220)),
    ("Yellow", (80, 230, 255)),
    ("Red", (80, 90, 255)),
    ("White", (255, 255, 255)),
)
RAINBOW_INDEX = len(COLORS)
PALETTE = tuple(color for _, color in COLORS[:5])


# --------------------------------------------------------------------------
# Hand measurements
# --------------------------------------------------------------------------

def palm_size(hand: np.ndarray) -> float:
    """Wrist to middle-finger knuckle, used as a ruler that scales with the hand."""
    return max(float(np.linalg.norm(hand[MIDDLE_MCP_ID] - hand[WRIST_ID])), 1.0)


def palm_center(hand: np.ndarray) -> np.ndarray:
    return hand[list(PALM_IDS)].mean(axis=0)


def finger_reach(hand: np.ndarray) -> float:
    """Distance of the most stretched finger from the wrist, in palm sizes.

    Roughly 2 for an open hand and 1 for a fist.
    """
    tips = hand[list(FINGERTIP_IDS)]
    return float(np.linalg.norm(tips - hand[WRIST_ID], axis=1).max() / palm_size(hand))


def finger_ratios(hand: np.ndarray) -> np.ndarray:
    """Tip-to-wrist over knuckle-to-wrist for index, middle, ring, pinky.

    Above about 1.2 a finger is stretched, below 1.0 it is curled.
    """
    wrist = hand[WRIST_ID]
    tips = np.linalg.norm(hand[list(FINGERTIP_IDS)] - wrist, axis=1)
    pips = np.linalg.norm(hand[list(FINGER_PIP_IDS)] - wrist, axis=1)
    return tips / np.maximum(pips, 1e-3)


def is_pointing(hand: np.ndarray) -> bool:
    """True for a hand that points with the index finger."""
    ratios = finger_ratios(hand)
    return bool(ratios[0] >= POINT_INDEX_MIN and int((ratios[1:] <= POINT_CURL_MAX).sum()) >= 2)


def dedupe_hands(hands):
    """MediaPipe sometimes reports one real hand twice. Keep only one of them,
    otherwise a single hand could "join its own fingertips" and start drawing."""
    kept: list[np.ndarray] = []
    for hand in hands:
        centre, size = palm_center(hand), palm_size(hand)
        if all(np.linalg.norm(centre - palm_center(other)) > 0.6 * min(size, palm_size(other))
               for other in kept):
            kept.append(hand)
    return kept


# --------------------------------------------------------------------------
# The pen: two index fingertips joined together
# --------------------------------------------------------------------------

class AirPen:
    """Draws while the index fingertips of two hands touch each other."""

    def __init__(self, join_ratio: float = 0.55, split_ratio: float | None = None,
                 lift_grace: float = 0.25, smoothing: float = 0.55):
        # Distances are in palm sizes. The pen goes down below join_ratio and
        # stays down until the tips are further apart than split_ratio.
        self.join_ratio = join_ratio
        self.split_ratio = split_ratio if split_ratio is not None else join_ratio * 1.7
        self.lift_grace = lift_grace
        self.smoothing = smoothing
        self.down = False
        self.points: list[tuple[float, float]] = []
        self.cursor: np.ndarray | None = None
        self.joined = False
        self._lost_since: float | None = None

    def reset(self) -> None:
        self.down = False
        self.points = []
        self.cursor = None
        self.joined = False
        self._lost_since = None

    @staticmethod
    def _closest_pair(hands):
        best = None
        for i in range(len(hands)):
            for j in range(i + 1, len(hands)):
                scale = (palm_size(hands[i]) + palm_size(hands[j])) / 2.0
                # Two detections of the same hand are not two hands.
                if np.linalg.norm(palm_center(hands[i]) - palm_center(hands[j])) < 0.6 * scale:
                    continue
                a, b = hands[i][INDEX_TIP_ID], hands[j][INDEX_TIP_ID]
                ratio = float(np.linalg.norm(a - b)) / scale
                if best is None or ratio < best[0]:
                    best = (ratio, (a + b) / 2.0)
        return best

    def update(self, hands, now: float, enabled: bool = True):
        """Feed the hands seen in one frame.

        Returns the finished stroke (a list of points) once, when the pen
        lifts. Returns None the rest of the time.
        """
        pair = self._closest_pair(hands) if enabled else None
        limit = self.split_ratio if self.down else self.join_ratio
        self.joined = pair is not None and pair[0] < limit

        if self.joined:
            self._lost_since = None
            position = pair[1].astype(np.float32)
            if not self.down:
                self.down = True
                self.cursor = position
                self.points = [(float(position[0]), float(position[1]))]
            else:
                self.cursor = self.cursor + self.smoothing * (position - self.cursor)
                last = self.points[-1]
                if math.hypot(self.cursor[0] - last[0], self.cursor[1] - last[1]) >= 2.0:
                    self.points.append((float(self.cursor[0]), float(self.cursor[1])))
            return None

        if not self.down:
            self.cursor = None
            return None
        # The tips came apart, or tracking blinked. Wait a moment before
        # deciding the stroke is over.
        if self._lost_since is None:
            self._lost_since = now
        if enabled and now - self._lost_since < self.lift_grace:
            return None
        stroke = self.points
        self.reset()
        return stroke


# --------------------------------------------------------------------------
# Shapes
# --------------------------------------------------------------------------

def rainbow_colors(count: int, offset: float = 0.0) -> np.ndarray:
    """(count, 3) BGR colours running through the rainbow."""
    count = max(int(count), 1)
    hue = ((offset + np.linspace(0.0, 1.0, count, endpoint=False)) % 1.0 * 179.0).astype(np.uint8)
    hsv = np.stack([hue, np.full(count, 235, np.uint8), np.full(count, 255, np.uint8)], axis=1)
    return cv2.cvtColor(hsv.reshape(1, count, 3), cv2.COLOR_HSV2BGR).reshape(count, 3)


@dataclass(eq=False)
class Shape:
    kind: str                       # line, circle, ellipse, triangle, square,
                                    # rectangle, quad or freehand
    outline: np.ndarray             # (N, 2) points relative to the center, never edited in place
    closed: bool
    center: np.ndarray
    color: tuple[int, int, int]
    radius: float                   # how far from the center a hand can catch it
    thickness: int = 3
    rainbow: bool = False
    colors: np.ndarray | None = None   # per-vertex colours for rainbow shapes
    scale: float = 1.0
    angle: float = 0.0              # radians
    held: bool = False
    grip: np.ndarray | None = None
    grip_offset: np.ndarray | None = None
    grip2: np.ndarray | None = None          # second hand while resizing / rotating
    grip2_lost_at: float | None = None
    pinch: dict | None = None
    grip_lost_at: float | None = None
    reach_at: float = float("-inf")
    dirty: bool = False

    def points(self) -> np.ndarray:
        if self.scale == 1.0 and self.angle == 0.0:
            return self.outline + self.center
        cos, sin = math.cos(self.angle), math.sin(self.angle)
        rotation = np.array([[cos, -sin], [sin, cos]], dtype=np.float32)
        return (self.outline * self.scale) @ rotation.T + self.center

    @property
    def reach_radius(self) -> float:
        return self.radius * self.scale

    def snapshot(self) -> "Shape":
        """A copy for the undo history (not held, no grip)."""
        return Shape(kind=self.kind, outline=self.outline, closed=self.closed,
                     center=self.center.copy(), color=self.color, radius=self.radius,
                     thickness=self.thickness, rainbow=self.rainbow, colors=self.colors,
                     scale=self.scale, angle=self.angle)


def _smooth(points: np.ndarray, window: int = 5) -> np.ndarray:
    pad = window // 2
    padded = np.pad(points, ((pad, pad), (0, 0)), mode="edge")
    kernel = np.ones(window) / window
    columns = [np.convolve(padded[:, axis], kernel, mode="valid") for axis in range(2)]
    return np.stack(columns, axis=1).astype(np.float32)


def _resample(points: np.ndarray, count: int) -> np.ndarray:
    steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    along = np.concatenate([[0.0], np.cumsum(steps)])
    if along[-1] < 1e-6:
        return np.repeat(points[:1], count, axis=0).astype(np.float32)
    targets = np.linspace(0.0, along[-1], count)
    x = np.interp(targets, along, points[:, 0])
    y = np.interp(targets, along, points[:, 1])
    return np.stack([x, y], axis=1).astype(np.float32)


def _open_stroke(points: np.ndarray, path: float):
    start, end = points[0], points[-1]
    chord = end - start
    length = float(np.linalg.norm(chord))
    if length > 0.8 * path:
        normal = np.array([-chord[1], chord[0]], dtype=np.float32) / length
        deviation = float(np.abs((points - start) @ normal).max())
        if deviation < 0.09 * length:
            return "line", np.stack([start, end]).astype(np.float32), None, False
    return "freehand", _resample(points, 64), None, False


def _ellipse_ring(cx, cy, w, h, angle):
    t = np.linspace(0.0, 2.0 * math.pi, 72, endpoint=False)
    a, b, theta = w / 2.0, h / 2.0, math.radians(angle)
    x, y = a * np.cos(t), b * np.sin(t)
    return np.stack([cx + x * math.cos(theta) - y * math.sin(theta),
                     cy + x * math.sin(theta) + y * math.cos(theta)], axis=1).astype(np.float32)


def _try_ellipse(points: np.ndarray, contour: np.ndarray):
    """Elongated ellipses are not round enough for the circularity test, so
    fit one and see how well the stroke follows it."""
    (cx, cy), (w, h), angle = cv2.fitEllipse(contour)
    if min(w, h) < 1.0:
        return None
    theta = math.radians(angle)
    dx, dy = points[:, 0] - cx, points[:, 1] - cy
    u = (dx * math.cos(theta) + dy * math.sin(theta)) / (w / 2.0)
    v = (-dx * math.sin(theta) + dy * math.cos(theta)) / (h / 2.0)
    error = float(np.abs(np.sqrt(u * u + v * v) - 1.0).mean())
    if error > 0.07:
        return None
    return "ellipse", _ellipse_ring(cx, cy, w, h, angle), np.array([cx, cy], dtype=np.float32), True


def _closed_stroke(points: np.ndarray):
    contour = points.reshape(-1, 1, 2).astype(np.float32)
    hull = cv2.convexHull(contour)
    area = float(cv2.contourArea(hull))
    perimeter = float(cv2.arcLength(hull, True))
    if area < 1.0 or perimeter < 1.0:
        return "freehand", _resample(points, 64), None, True

    circularity = 4.0 * math.pi * area / (perimeter * perimeter)
    if circularity >= 0.87:
        (cx, cy), (w, h), angle = cv2.fitEllipse(contour)
        rounded = min(w, h) / max(w, h, 1e-6) > 0.82
        if rounded:
            w = h = (w + h) / 2.0
            angle = 0.0
        ring = _ellipse_ring(cx, cy, w, h, angle)
        return ("circle" if rounded else "ellipse"), ring, \
            np.array([cx, cy], dtype=np.float32), True

    corners = cv2.approxPolyDP(hull, 0.035 * perimeter, True).reshape(-1, 2).astype(np.float32)
    if len(corners) == 3:
        return "triangle", corners, None, True
    if len(corners) == 4:
        rect = cv2.minAreaRect(hull)
        (rx, ry), (rw, rh), angle = rect
        if rw * rh > 0 and area / (rw * rh) > 0.8:
            box = cv2.boxPoints(rect)
            tilt = min(angle % 90.0, 90.0 - angle % 90.0)
            if tilt < 10.0:
                # Close enough to upright: straighten it.
                x0, y0 = box.min(axis=0)
                x1, y1 = box.max(axis=0)
                box = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)
            width = float(np.linalg.norm(box[1] - box[0]))
            height = float(np.linalg.norm(box[2] - box[1]))
            kind = "rectangle"
            if abs(width - height) / max(width, height, 1e-6) < 0.12:
                kind = "square"
                side = (width + height) / 2.0
                middle = box.mean(axis=0)
                direction = (box[1] - box[0]) / max(width, 1e-6)
                across = np.array([-direction[1], direction[0]], dtype=np.float32)
                box = np.array([
                    middle - direction * side / 2 - across * side / 2,
                    middle + direction * side / 2 - across * side / 2,
                    middle + direction * side / 2 + across * side / 2,
                    middle - direction * side / 2 + across * side / 2,
                ], dtype=np.float32)
            return kind, box.astype(np.float32), None, True
        elongated = _try_ellipse(points, contour)
        return elongated if elongated else ("quad", corners, None, True)
    elongated = _try_ellipse(points, contour) if len(corners) >= 5 else None
    return elongated if elongated else ("freehand", _resample(points, 64), None, True)


def recognize_shape(points, color: tuple[int, int, int] = (255, 255, 255),
                    thickness: int = 3, rainbow: bool = False) -> Shape | None:
    """Turn a drawn path into a clean Shape, or None if it is too small."""
    if points is None or len(points) < MIN_STROKE_POINTS:
        return None
    path_points = _smooth(np.asarray(points, dtype=np.float32))
    steps = np.linalg.norm(np.diff(path_points, axis=0), axis=1)
    path = float(steps.sum())
    low, high = path_points.min(axis=0), path_points.max(axis=0)
    diagonal = float(np.linalg.norm(high - low))
    if diagonal < MIN_SHAPE_SIZE:
        return None

    gap = float(np.linalg.norm(path_points[0] - path_points[-1]))
    closed = gap < 0.35 * diagonal and path > 1.7 * diagonal
    if closed:
        kind, outline, center, is_closed = _closed_stroke(path_points)
    else:
        kind, outline, center, is_closed = _open_stroke(path_points, path)

    if center is None:
        if kind in ("triangle", "square", "rectangle", "quad"):
            center = outline.mean(axis=0)
        else:
            center = (outline.min(axis=0) + outline.max(axis=0)) / 2.0
    center = np.asarray(center, dtype=np.float32)
    outline = (outline - center).astype(np.float32)
    radius = max(float(np.linalg.norm(outline, axis=1).max()), 35.0)
    colors = rainbow_colors(len(outline)) if rainbow else None
    return Shape(kind=kind, outline=outline, closed=is_closed, center=center,
                 color=color, radius=radius, thickness=int(thickness),
                 rainbow=rainbow, colors=colors)


def polyline_distance(point, points: np.ndarray, closed: bool) -> float:
    """Shortest distance from a point to a polyline."""
    pts = np.asarray(points, dtype=np.float32)
    if closed and len(pts) > 2:
        pts = np.vstack([pts, pts[:1]])
    if len(pts) < 2:
        return float(np.linalg.norm(np.asarray(point, dtype=np.float32) - pts[0]))
    a, b = pts[:-1], pts[1:]
    ab = b - a
    p = np.asarray(point, dtype=np.float32)
    t = np.clip(((p - a) * ab).sum(axis=1) / np.maximum((ab * ab).sum(axis=1), 1e-9), 0.0, 1.0)
    nearest = a + ab * t[:, None]
    return float(np.linalg.norm(p - nearest, axis=1).min())


def _draw_poly(image, pts, closed, color, thickness, colors=None) -> None:
    """Draw one outline. Rainbow shapes pass per-vertex colours."""
    if colors is None:
        cv2.polylines(image, [pts.reshape(-1, 1, 2)], closed, color, thickness, cv2.LINE_AA)
        return
    count = len(pts)
    segments = count if closed else count - 1
    points = pts.tolist()
    for i in range(segments):
        c = colors[i % len(colors)]
        cv2.line(image, tuple(points[i]), tuple(points[(i + 1) % count]),
                 (int(c[0]), int(c[1]), int(c[2])), thickness, cv2.LINE_AA)


# --------------------------------------------------------------------------
# The board that holds the shapes
# --------------------------------------------------------------------------

class ShapeBoard:
    """Stores shapes, colours and history; lets hands carry, resize, rotate and
    erase shapes; paints them over the video."""

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.shapes: list[Shape] = []
        self._colors_used = 0
        self.color_index: int | None = None      # None = change colour per shape
        self.thickness = 3
        self._history: list[list[Shape]] = [[]]
        self._cursor = 0
        self._erase_pending: float | None = None
        self._need_commit = False

    # -- colours and pen ---------------------------------------------------

    @property
    def holding(self) -> bool:
        return any(shape.held for shape in self.shapes)

    @property
    def manipulating(self) -> bool:
        return any(shape.held and shape.pinch is not None for shape in self.shapes)

    def set_color(self, index: int | None) -> None:
        if index is not None and not 0 <= index <= RAINBOW_INDEX:
            raise ValueError("colour index out of range")
        self.color_index = index

    def color_name(self) -> str:
        if self.color_index is None:
            return "Auto"
        return "Rainbow" if self.color_index == RAINBOW_INDEX else COLORS[self.color_index][0]

    @property
    def rainbow(self) -> bool:
        return self.color_index == RAINBOW_INDEX

    def next_color(self) -> tuple[int, int, int]:
        """The colour the next shape gets (also used for the stroke being drawn)."""
        if self.color_index is None:
            return PALETTE[self._colors_used % len(PALETTE)]
        if self.color_index == RAINBOW_INDEX:
            return (255, 255, 255)
        return COLORS[self.color_index][1]

    def change_thickness(self, delta: int) -> int:
        self.thickness = int(np.clip(self.thickness + delta, 2, 10))
        return self.thickness

    # -- history -----------------------------------------------------------

    def _commit(self) -> None:
        snapshot = [shape.snapshot() for shape in self.shapes]
        del self._history[self._cursor + 1:]
        self._history.append(snapshot)
        if len(self._history) > HISTORY_LIMIT:
            del self._history[0]
        self._cursor = len(self._history) - 1

    def flush(self) -> None:
        """Close any open history step (a finished eraser sweep, a released shape)."""
        self._release_all()
        if self._erase_pending is not None:
            self._erase_pending = None
            self._commit()

    def tick(self, now: float) -> None:
        if self._erase_pending is not None and now - self._erase_pending >= ERASE_COMMIT_DELAY:
            self._erase_pending = None
            self._commit()

    @property
    def can_undo(self) -> bool:
        return self._cursor > 0 or self._erase_pending is not None or self.holding

    @property
    def can_redo(self) -> bool:
        return self._cursor < len(self._history) - 1

    def undo(self) -> bool:
        self.flush()
        if self._cursor == 0:
            return False
        self._cursor -= 1
        self.shapes = [shape.snapshot() for shape in self._history[self._cursor]]
        return True

    def redo(self) -> bool:
        self.flush()
        if self._cursor >= len(self._history) - 1:
            return False
        self._cursor += 1
        self.shapes = [shape.snapshot() for shape in self._history[self._cursor]]
        return True

    # -- editing -----------------------------------------------------------

    def add(self, shape: Shape) -> None:
        self.flush()
        self.shapes.append(shape)
        self._colors_used += 1
        while len(self.shapes) > MAX_SHAPES:
            del self.shapes[0]
        self._commit()

    def clear(self) -> bool:
        self.flush()
        if not self.shapes:
            return False
        self.shapes.clear()
        self._commit()
        return True

    def _touches(self, shape: Shape, point, radius: float) -> bool:
        pts = shape.points()
        if shape.closed and len(pts) >= 3 and cv2.pointPolygonTest(
                pts.reshape(-1, 1, 2).astype(np.float32), (float(point[0]), float(point[1])), False) >= 0:
            return True
        return polyline_distance(point, pts, shape.closed) <= radius + shape.thickness

    def erase_at(self, points, radius: float, now: float) -> int:
        """Remove every shape touched by one of the points. Returns how many."""
        if not points:
            return 0
        kept = [shape for shape in self.shapes
                if shape.held or not any(self._touches(shape, p, radius) for p in points)]
        removed = len(self.shapes) - len(kept)
        if removed:
            self.shapes[:] = kept
            self._erase_pending = now
        return removed

    def rescale(self, width: int, height: int) -> None:
        """The camera changed its picture size: keep the shapes where they were."""
        if (width, height) == (self.width, self.height):
            return
        fx, fy = width / self.width, height / self.height
        factor = (fx + fy) / 2.0
        for history in [self.shapes] + self._history:
            for shape in history:
                shape.center = shape.center * np.array([fx, fy], dtype=np.float32)
        seen: set[int] = set()
        for history in [self.shapes] + self._history:
            for shape in history:
                if id(shape.outline) not in seen:
                    seen.add(id(shape.outline))
                    shape.outline *= factor
                    shape.radius *= factor
        self.width, self.height = width, height

    # -- hands -------------------------------------------------------------

    def _clamp(self, point: np.ndarray) -> np.ndarray:
        return np.clip(point, [0.0, 0.0], [self.width - 1.0, self.height - 1.0]).astype(np.float32)

    def _drop(self, shape: Shape) -> None:
        if shape.dirty:
            self._need_commit = True
        shape.held = False
        shape.grip = shape.grip_offset = shape.grip2 = shape.pinch = None
        shape.grip_lost_at = shape.grip2_lost_at = None
        shape.dirty = False

    def _release_all(self) -> None:
        for shape in self.shapes:
            if shape.held:
                self._drop(shape)
        if self._need_commit:
            self._need_commit = False
            self._commit()

    @staticmethod
    def _nearest(anchor, palms, reach, used, limit, max_reach=FIST_OPEN):
        best = None
        for i, (palm, r) in enumerate(zip(palms, reach)):
            if i in used or r >= max_reach:
                continue
            distance = float(np.linalg.norm(palm - anchor))
            if distance <= limit and (best is None or distance < best[0]):
                best = (distance, i)
        return None if best is None else best[1]

    @staticmethod
    def _end_pinch(shape: Shape, palm: np.ndarray) -> None:
        shape.grip2 = shape.pinch = shape.grip2_lost_at = None
        shape.grip = palm
        shape.grip_offset = shape.center - palm

    def _start_pinch(self, shape: Shape, p1: np.ndarray, p2: np.ndarray) -> None:
        vector = p2 - p1
        mid = (p1 + p2) / 2.0
        shape.grip2 = p2
        shape.grip2_lost_at = None
        shape.pinch = {
            "d0": max(float(np.linalg.norm(vector)), 1.0),
            "a0": math.atan2(float(vector[1]), float(vector[0])),
            "s0": shape.scale, "r0": shape.angle,
            "off0": shape.center - mid,
        }

    def _apply_pinch(self, shape: Shape, p1: np.ndarray, p2: np.ndarray) -> None:
        pinch = shape.pinch
        vector = p2 - p1
        distance = max(float(np.linalg.norm(vector)), 1.0)
        turn = math.atan2(float(vector[1]), float(vector[0])) - pinch["a0"]
        scale = float(np.clip(pinch["s0"] * distance / pinch["d0"], MIN_SCALE, MAX_SCALE))
        ratio = scale / pinch["s0"]
        cos, sin = math.cos(turn), math.sin(turn)
        offset = pinch["off0"] * ratio
        rotated = np.array([offset[0] * cos - offset[1] * sin,
                            offset[0] * sin + offset[1] * cos], dtype=np.float32)
        shape.scale = scale
        shape.angle = pinch["r0"] + turn
        shape.center = self._clamp((p1 + p2) / 2.0 + rotated)
        shape.grip, shape.grip2 = p1, p2
        shape.dirty = True

    def _assign(self, shape, palms, reach, used):
        """Which closed hand holds the first grip and which the second."""
        pool = [i for i in range(len(palms)) if i not in used and reach[i] < FIST_OPEN]
        if shape.grip2 is None:
            i1 = self._nearest(shape.grip, palms, reach, used, FOLLOW_DISTANCE)
            return i1, None
        best = None
        for a in pool:
            for b in pool:
                if a == b:
                    continue
                da = float(np.linalg.norm(palms[a] - shape.grip))
                db = float(np.linalg.norm(palms[b] - shape.grip2))
                if da <= FOLLOW_DISTANCE and db <= FOLLOW_DISTANCE and (best is None or da + db < best[0]):
                    best = (da + db, a, b)
        if best is not None:
            return best[1], best[2]
        # One hand only: it belongs to whichever grip it is nearest to.
        single = None
        for c in pool:
            da = float(np.linalg.norm(palms[c] - shape.grip))
            db = float(np.linalg.norm(palms[c] - shape.grip2))
            d = min(da, db)
            if d <= FOLLOW_DISTANCE and (single is None or d < single[0]):
                single = (d, c, da <= db)
        if single is None:
            return None, None
        return (single[1], None) if single[2] else (None, single[1])

    def _follow(self, shape, palms, sizes, reach, used, now) -> None:
        i1, i2 = self._assign(shape, palms, reach, used)
        if i1 is not None:
            used.add(i1)
        if i2 is not None:
            used.add(i2)

        if i1 is None and i2 is None:
            if shape.grip_lost_at is None:
                shape.grip_lost_at = now
            elif now - shape.grip_lost_at > RELEASE_GRACE:
                self._drop(shape)
            return
        shape.grip_lost_at = None
        if i1 is None:                      # the first hand left, the second carries on alone
            i1, i2 = i2, None
            self._end_pinch(shape, palms[i1])

        if shape.grip2 is not None:
            if i2 is None:                  # second hand blinked out or opened
                if shape.grip2_lost_at is None:
                    shape.grip2_lost_at = now
                elif now - shape.grip2_lost_at > RELEASE_GRACE:
                    self._end_pinch(shape, palms[i1])
                return
            shape.grip2_lost_at = None
            self._apply_pinch(shape, palms[i1], palms[i2])
            return

        # One hand carries the shape.
        palm = palms[i1]
        new_center = self._clamp(palm + shape.grip_offset)
        if float(np.linalg.norm(new_center - shape.center)) > 0.5:
            shape.dirty = True
        shape.grip = palm
        shape.center = new_center
        # A second closed hand somewhere else starts the resize / rotate.
        other = self._nearest(palm, palms, reach, used, float("inf"), FIST_CLOSE)
        if other is not None and float(np.linalg.norm(palms[other] - palm)) >= \
                1.2 * max(sizes[i1], sizes[other]):
            used.add(other)
            self._start_pinch(shape, palm, palms[other])

    def update_hands(self, hands, now: float, allow_grab: bool = True) -> None:
        """Move shapes that are in a closed hand and catch new ones."""
        self.tick(now)
        palms = [palm_center(h) for h in hands]
        sizes = [palm_size(h) for h in hands]
        reach = [finger_reach(h) for h in hands]
        used: set[int] = set()

        # 1. Shapes that are already being carried follow their hands.
        for shape in self.shapes:
            if shape.held:
                self._follow(shape, palms, sizes, reach, used, now)

        # 2. An open hand held over a shape is the "reach" before a catch.
        for palm, r in zip(palms, reach):
            if r < FIST_OPEN:
                continue
            for shape in self.shapes:
                if not shape.held and \
                        np.linalg.norm(palm - shape.center) <= shape.reach_radius + GRAB_MARGIN:
                    shape.reach_at = now

        # 3. A hand that closes over a shape catches it.
        if allow_grab:
            for i, (palm, r) in enumerate(zip(palms, reach)):
                if i in used or r >= FIST_CLOSE:
                    continue
                best = None
                for shape in self.shapes:
                    if shape.held or now - shape.reach_at > REACH_WINDOW:
                        continue
                    distance = float(np.linalg.norm(palm - shape.center))
                    if distance <= shape.reach_radius + GRAB_MARGIN and \
                            (best is None or distance < best[0]):
                        best = (distance, shape)
                if best is None:
                    continue
                shape = best[1]
                shape.held = True
                shape.dirty = False
                shape.grip = palm
                shape.grip_offset = shape.center - palm
                shape.grip_lost_at = None
                used.add(i)
                # The shape you just caught is painted on top.
                self.shapes[:] = [s for s in self.shapes if s is not shape]
                self.shapes.append(shape)

        if self._need_commit:
            self._need_commit = False
            self._commit()

    # -- painting ----------------------------------------------------------

    def draw(self, canvas: np.ndarray, stroke=None,
             stroke_color: tuple[int, int, int] = (255, 255, 255),
             stroke_rainbow: bool = False) -> np.ndarray:
        """Paint shapes (and the stroke being drawn) onto the canvas, in place.

        The glow is made on a half-size layer covering only the area that has
        something in it, so the cost does not grow with the picture size.
        """
        items = []
        for shape in self.shapes:
            thick = shape.thickness + (2 if shape.held else 0)
            items.append((np.round(shape.points()).astype(np.int32), shape.closed, shape.color,
                          shape.colors if shape.rainbow else None, thick, shape.closed))
        if stroke is not None and len(stroke) >= 2:
            pts = np.round(np.asarray(stroke, dtype=np.float32)).astype(np.int32)
            colors = rainbow_colors(len(pts)) if stroke_rainbow else None
            items.append((pts, False, stroke_color, colors, self.thickness, False))
        if not items:
            return canvas

        height, width = canvas.shape[:2]
        # Each shape glows inside its own small box, so shapes scattered over the
        # picture never cost a pass over the whole picture.
        for pts, closed, color, colors, thick, filled in items:
            margin = 14 + 2 * thick
            x0, y0 = (int(v) for v in np.maximum(pts.min(axis=0) - margin, 0))
            x1 = min(int(pts[:, 0].max()) + margin, width)
            y1 = min(int(pts[:, 1].max()) + margin, height)
            if x1 - x0 < 4 or y1 - y0 < 4:
                continue
            roi = canvas[y0:y1, x0:x1]
            roi_h, roi_w = roi.shape[:2]
            half_h, half_w = max(1, roi_h // 2), max(1, roi_w // 2)
            small = np.round((pts - np.array([x0, y0], dtype=np.float32)) / 2.0).astype(np.int32)
            glow = np.zeros((half_h, half_w, 3), np.uint8)
            _draw_poly(glow, small, closed, color, max(2, thick + 3), colors)
            glow = cv2.GaussianBlur(glow, (0, 0), 3.0)
            glow = cv2.resize(glow, (roi_w, roi_h), interpolation=cv2.INTER_LINEAR)
            roi[:] = cv2.addWeighted(roi, 1.0, glow, 0.85, 0)
            if filled:
                fill = np.zeros((half_h, half_w, 3), np.uint8)
                fill_mask = np.zeros((half_h, half_w), np.uint8)
                cv2.fillPoly(fill, [small.reshape(-1, 1, 2)], color)
                cv2.fillPoly(fill_mask, [small.reshape(-1, 1, 2)], 255)
                alpha = cv2.resize(fill_mask, (roi_w, roi_h), interpolation=cv2.INTER_LINEAR)
                alpha = alpha.astype(np.float32) * (0.26 / 255.0)
                fill = cv2.resize(fill, (roi_w, roi_h), interpolation=cv2.INTER_LINEAR)
                roi[:] = cv2.blendLinear(roi, fill, 1.0 - alpha, alpha)

        for pts, closed, color, colors, thick, _ in items:
            _draw_poly(canvas, pts, closed, color, thick, colors)
            _draw_poly(canvas, pts, closed, (255, 255, 255), 1)
        return canvas

    def render(self, frame: np.ndarray, stroke=None,
               stroke_color: tuple[int, int, int] = (255, 255, 255),
               stroke_rainbow: bool = False) -> np.ndarray:
        """Like draw(), but returns a new image when something is drawn, so the
        input (for example the saved background) is never painted on."""
        has_stroke = stroke is not None and len(stroke) >= 2
        if not self.shapes and not has_stroke:
            return frame
        out = frame.copy()
        self.draw(out, stroke, stroke_color, stroke_rainbow)
        return out
