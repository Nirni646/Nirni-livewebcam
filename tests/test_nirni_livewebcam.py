"""Unit tests for Nirni Livewebcam. Run with:  python -m unittest discover -s tests

Copyright (c) 2026 Sidagam Nirni Naidu. All rights reserved.
"""

import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import air_canvas as ac  # noqa: E402
import effects as fx  # noqa: E402
import gestures as gs  # noqa: E402
import nirni_livewebcam as nv  # noqa: E402
import tracking as tr  # noqa: E402


NOSE = (400.0, 300.0)
FACE_WIDTH = 200.0


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def make_hand(center=(300.0, 200.0), size=100.0, fist=False, index_tip=None):
    """A made-up hand with 21 landmarks. Palm size equals `size`."""
    cx, cy = center
    hand = np.zeros((21, 2), np.float32)
    hand[0] = (cx, cy + size)                       # wrist
    hand[9] = (cx, cy)                              # middle knuckle
    knuckles = {5: (cx - 0.45 * size, cy + 0.05 * size),
                13: (cx + 0.40 * size, cy + 0.08 * size),
                17: (cx + 0.75 * size, cy + 0.20 * size)}
    for i, point in knuckles.items():
        hand[i] = point
    for tip, knuckle in ((8, 5), (12, 9), (16, 13), (20, 17)):
        kx, ky = hand[knuckle]
        hand[tip] = (kx, cy + 0.15 * size) if fist else (kx, ky - 0.95 * size)
    if index_tip is not None:
        hand[8] = index_tip
    return hand


def circle_stroke(center=(300, 200), radius=80, turns=1.05, count=60, noise=1.5, seed=3):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 2 * math.pi * turns, count)
    pts = np.stack([center[0] + radius * np.cos(t), center[1] + radius * np.sin(t)], axis=1)
    return pts + rng.normal(0, noise, pts.shape)


def polygon_stroke(corners, per_edge=18, noise=1.2, seed=5):
    rng = np.random.default_rng(seed)
    pts = []
    loop = list(corners) + [corners[0]]
    for a, b in zip(loop[:-1], loop[1:]):
        for k in range(per_edge):
            pts.append(np.array(a) + (np.array(b) - np.array(a)) * k / per_edge)
    pts.append(np.array(corners[0], dtype=float) + np.array([3.0, 2.0]))
    pts = np.array(pts, dtype=float)
    return pts + rng.normal(0, noise, pts.shape)


# --------------------------------------------------------------------------
# Touch detection
# --------------------------------------------------------------------------

class TouchDetectorTests(unittest.TestCase):
    def make(self, **kwargs):
        return gs.TouchDetector(hold_seconds=0.4, distance_ratio=0.22, **kwargs)

    def test_touch_must_be_held(self):
        d = self.make()
        self.assertFalse(d.update(NOSE, FACE_WIDTH, [(405, 305)], 0.0))
        self.assertFalse(d.update(NOSE, FACE_WIDTH, [(405, 305)], 0.2))
        self.assertTrue(d.update(NOSE, FACE_WIDTH, [(405, 305)], 0.45))

    def test_far_finger_never_triggers(self):
        d = self.make()
        for t in np.arange(0, 3, 0.05):
            self.assertFalse(d.update(NOSE, FACE_WIDTH, [(600, 300)], float(t)))

    def test_no_hand_never_triggers(self):
        d = self.make()
        for t in np.arange(0, 3, 0.05):
            self.assertFalse(d.update(NOSE, FACE_WIDTH, [], float(t)))

    def test_brief_touch_resets(self):
        d = self.make()
        d.update(NOSE, FACE_WIDTH, [(402, 300)], 0.0)
        d.update(NOSE, FACE_WIDTH, [(402, 300)], 0.3)
        d.update(NOSE, FACE_WIDTH, [(700, 300)], 0.35)   # finger leaves
        self.assertFalse(d.update(NOSE, FACE_WIDTH, [(402, 300)], 0.4))
        self.assertFalse(d.update(NOSE, FACE_WIDTH, [(402, 300)], 0.7))
        self.assertTrue(d.update(NOSE, FACE_WIDTH, [(402, 300)], 0.85))

    def test_fires_once_until_finger_leaves(self):
        d = self.make()
        fired = [d.update(NOSE, FACE_WIDTH, [(401, 301)], t) for t in np.arange(0, 3, 0.05)]
        self.assertEqual(sum(fired), 1)
        d.update(NOSE, FACE_WIDTH, [(700, 300)], 3.0)    # release
        fired = [d.update(NOSE, FACE_WIDTH, [(401, 301)], 3.1 + t) for t in np.arange(0, 1, 0.05)]
        self.assertEqual(sum(fired), 1)

    def test_disarm_requires_release(self):
        d = self.make()
        d.disarm()
        fired = [d.update(NOSE, FACE_WIDTH, [(401, 301)], t) for t in np.arange(0, 2, 0.05)]
        self.assertEqual(sum(fired), 0)

    def test_target_remembered_when_face_blinks_out(self):
        d = self.make()
        d.update(NOSE, FACE_WIDTH, [(401, 301)], 0.0)
        # the face tracker loses the face (finger in front of it)
        self.assertFalse(d.update(None, 0.0, [(401, 301)], 0.2))
        self.assertTrue(d.update(None, 0.0, [(401, 301)], 0.45))

    def test_target_memory_expires(self):
        d = self.make()
        d.update(NOSE, FACE_WIDTH, [], 0.0)
        for t in (1.0, 1.5, 2.0, 2.5):
            self.assertFalse(d.update(None, 0.0, [(401, 301)], t))

    def test_distance_scales_with_face_width(self):
        d = self.make()
        d.update(NOSE, 400.0, [(440, 300)], 0.0)
        self.assertLessEqual(d.closest_ratio, 0.22)
        d2 = self.make()
        d2.update(NOSE, 100.0, [(440, 300)], 0.0)
        self.assertGreater(d2.closest_ratio, 0.22)

    def test_either_hand_counts(self):
        d = self.make()
        tips = [(900, 100), (399, 298)]
        d.update(NOSE, FACE_WIDTH, tips, 0.0)
        self.assertTrue(d.update(NOSE, FACE_WIDTH, tips, 0.5))

    def test_nose_and_forehead_are_separate(self):
        forehead = (400.0, 190.0)       # about 0.55 face widths above the nose
        nose_d = gs.TouchDetector(0.4, 0.22)
        forehead_d = gs.TouchDetector(0.4, 0.20)
        tips = [(400, 192)]
        nose_d.update(NOSE, FACE_WIDTH, tips, 0.0)
        forehead_d.update(forehead, FACE_WIDTH, tips, 0.0)
        self.assertGreater(nose_d.closest_ratio, 0.22)
        self.assertLessEqual(forehead_d.closest_ratio, 0.20)


class DetectorBuildTests(unittest.TestCase):
    """The detector with made-up model results, no camera and no models."""

    W, H = 1000, 600

    def results(self, fingertip, blends=None, extra_hand=None):
        face_points = [SimpleNamespace(x=0.5, y=0.5) for _ in range(468)]
        for i in tr.NOSE_TIP_IDS:
            face_points[i] = SimpleNamespace(x=0.5, y=0.55)
        for i in tr.FOREHEAD_IDS:
            face_points[i] = SimpleNamespace(x=0.5, y=0.30)
        face_points[tr.CHIN_IDS[0]] = SimpleNamespace(x=0.5, y=0.75)
        face_points[tr.FACE_LEFT_ID] = SimpleNamespace(x=0.4, y=0.5)
        face_points[tr.FACE_RIGHT_ID] = SimpleNamespace(x=0.6, y=0.5)
        hand_points = [SimpleNamespace(x=0.9, y=0.9) for _ in range(21)]
        hand_points[ac.INDEX_TIP_ID] = SimpleNamespace(x=fingertip[0], y=fingertip[1])
        hands = [hand_points] + ([extra_hand] if extra_hand else [])
        categories = [SimpleNamespace(category_name=k, score=v) for k, v in (blends or {}).items()]
        face = SimpleNamespace(face_landmarks=[face_points], face_blendshapes=[categories] if blends else [])
        return face, SimpleNamespace(hand_landmarks=hands)

    def detector(self):
        return tr.Detector(None, None)

    def test_positions_are_in_pixels(self):
        face, hand = self.results((0.5, 0.31))
        obs = self.detector().build(face, hand, self.W, self.H, frame_id=7)
        self.assertTrue(obs.face_found)
        self.assertEqual(obs.frame_id, 7)
        self.assertAlmostEqual(obs.nose[1], 0.55 * self.H)
        self.assertAlmostEqual(obs.forehead[1], 0.30 * self.H)
        self.assertAlmostEqual(obs.chin[1], 0.75 * self.H)
        self.assertAlmostEqual(obs.face_width, 0.2 * self.W)
        self.assertEqual(len(obs.hands), 1)
        self.assertEqual(obs.hands[0].shape, (21, 2))

    def test_blendshapes_are_read(self):
        face, hand = self.results((0.5, 0.3), {"eyeBlinkLeft": 0.9, "eyeBlinkRight": 0.1,
                                               "mouthSmileLeft": 0.6, "mouthSmileRight": 0.8,
                                               "jawOpen": 0.2})
        obs = self.detector().build(face, hand, self.W, self.H)
        self.assertAlmostEqual(obs.blink_left, 0.9)
        self.assertAlmostEqual(obs.blink_right, 0.1)
        self.assertAlmostEqual(obs.smile, 0.7)
        self.assertAlmostEqual(obs.jaw_open, 0.2)

    def test_missing_blendshapes_are_none(self):
        face, hand = self.results((0.5, 0.3))
        obs = self.detector().build(face, hand, self.W, self.H)
        self.assertIsNone(obs.smile)
        self.assertIsNone(obs.blink_left)

    def test_duplicate_hand_detection_is_dropped(self):
        twin = [SimpleNamespace(x=0.9005, y=0.9005) for _ in range(21)]
        face, hand = self.results((0.5, 0.3), extra_hand=twin)
        obs = self.detector().build(face, hand, self.W, self.H)
        self.assertEqual(len(obs.hands), 1)

    def test_timestamps_always_rise(self):
        detector = self.detector()
        values = [detector._timestamp(t) for t in (100, 100, 100, 90, 500)]
        self.assertEqual(values, sorted(set(values)))


class GestureEngineTests(unittest.TestCase):
    def obs(self, fingertip, **extra):
        hand = make_hand(index_tip=fingertip)
        base = dict(hands=[hand], face_found=True, nose=NOSE, forehead=(400.0, 200.0),
                    chin=(400.0, 420.0), face_width=FACE_WIDTH)
        base.update(extra)
        return tr.Observation(**base)

    def run_for(self, engine, obs, seconds=1.0, suppress=False):
        hits = []
        for t in np.arange(0.0, seconds, 0.05):
            hits.append(engine.process(obs, float(t), suppress=suppress))
        return hits

    def test_nose_forehead_and_chin_are_separate(self):
        for tip, name in (((401, 301), "nose"), ((400, 201), "forehead"), ((400, 421), "chin")):
            engine = gs.GestureEngine(hold=0.3)
            hits = self.run_for(engine, self.obs(tip))
            fired = [n for e in hits for n in ("nose", "forehead", "chin") if getattr(e, n)]
            self.assertEqual(fired, [name], name)

    def test_suppress_blocks_touches_until_release(self):
        engine = gs.GestureEngine(hold=0.3)
        hits = self.run_for(engine, self.obs((400, 201)), suppress=True)
        self.assertFalse(any(e.forehead for e in hits))
        hits = self.run_for(engine, self.obs((400, 201)))       # still resting there
        self.assertFalse(any(e.forehead for e in hits))

    def test_face_gestures_can_be_switched_off(self):
        engine = gs.GestureEngine(face_gestures=False)
        obs = self.obs((900, 500), blink_left=0.9, blink_right=0.0, smile=0.9, jaw_open=0.0)
        hits = self.run_for(engine, obs, seconds=2.0)
        self.assertFalse(any(e.wink or e.smile for e in hits))


# --------------------------------------------------------------------------
# Pen
# --------------------------------------------------------------------------

class AirPenTests(unittest.TestCase):
    def joined_hands(self, point):
        x, y = point
        return [make_hand((x - 120, y + 60), index_tip=(x - 3, y)),
                make_hand((x + 120, y + 60), index_tip=(x + 3, y))]

    def apart_hands(self):
        return [make_hand((200, 300), index_tip=(150, 200)),
                make_hand((700, 300), index_tip=(750, 200))]

    def draw_circle(self, pen, start=0.0):
        t = start
        for p in circle_stroke(noise=0.0, count=45):
            self.assertIsNone(pen.update(self.joined_hands(p), t))
            t += 0.03
        return t

    def test_stroke_is_returned_when_hands_part(self):
        pen = ac.AirPen()
        t = self.draw_circle(pen)
        self.assertTrue(pen.down)
        stroke = None
        for _ in range(20):
            t += 0.03
            stroke = pen.update(self.apart_hands(), t) or stroke
        self.assertIsNotNone(stroke)
        self.assertGreater(len(stroke), 20)
        self.assertFalse(pen.down)

    def test_short_dropout_does_not_end_the_stroke(self):
        pen = ac.AirPen()
        t = self.draw_circle(pen)
        t += 0.03
        self.assertIsNone(pen.update([], t))             # tracking blinks
        t += 0.1
        self.assertIsNone(pen.update(self.joined_hands((300, 280)), t))
        self.assertTrue(pen.down)

    def test_one_hand_never_draws(self):
        pen = ac.AirPen()
        for k in range(30):
            self.assertIsNone(pen.update([make_hand()], k * 0.03))
        self.assertFalse(pen.down)

    def test_disabled_pen_ignores_hands(self):
        pen = ac.AirPen()
        for k in range(10):
            pen.update(self.joined_hands((300, 200)), k * 0.03, enabled=False)
        self.assertFalse(pen.down)

    def test_switching_off_ends_the_stroke(self):
        pen = ac.AirPen()
        t = self.draw_circle(pen)
        stroke = pen.update(self.joined_hands((300, 200)), t + 0.03, enabled=False)
        self.assertIsNotNone(stroke)
        self.assertFalse(pen.down)


# --------------------------------------------------------------------------
# Shape recognition
# --------------------------------------------------------------------------

class RecognizeShapeTests(unittest.TestCase):
    def test_circle(self):
        shape = ac.recognize_shape(circle_stroke())
        self.assertEqual(shape.kind, "circle")
        self.assertTrue(shape.closed)
        self.assertAlmostEqual(shape.center[0], 300, delta=8)
        self.assertAlmostEqual(shape.center[1], 200, delta=8)

    def test_ellipse(self):
        rng = np.random.default_rng(2)
        t = np.linspace(0, 2 * math.pi * 1.05, 70)
        pts = np.stack([300 + 130 * np.cos(t), 200 + 55 * np.sin(t)], axis=1)
        shape = ac.recognize_shape(pts + rng.normal(0, 1.2, pts.shape))
        self.assertEqual(shape.kind, "ellipse")

    def test_square(self):
        shape = ac.recognize_shape(polygon_stroke([(200, 100), (350, 100), (350, 250), (200, 250)]))
        self.assertEqual(shape.kind, "square")
        self.assertEqual(len(shape.outline), 4)

    def test_rectangle(self):
        shape = ac.recognize_shape(polygon_stroke([(200, 100), (420, 100), (420, 200), (200, 200)]))
        self.assertEqual(shape.kind, "rectangle")

    def test_tilted_square_stays_a_square(self):
        shape = ac.recognize_shape(polygon_stroke([(300, 100), (400, 200), (300, 300), (200, 200)]))
        self.assertEqual(shape.kind, "square")

    def test_triangle(self):
        shape = ac.recognize_shape(polygon_stroke([(300, 100), (400, 260), (200, 260)]))
        self.assertEqual(shape.kind, "triangle")
        self.assertEqual(len(shape.outline), 3)

    def test_straight_line(self):
        rng = np.random.default_rng(4)
        t = np.linspace(0, 1, 40)[:, None]
        pts = np.array([100.0, 100.0]) + t * np.array([300.0, 50.0])
        shape = ac.recognize_shape(pts + rng.normal(0, 1.2, pts.shape))
        self.assertEqual(shape.kind, "line")
        self.assertFalse(shape.closed)

    def test_wavy_stroke_stays_freehand(self):
        x = np.linspace(100, 500, 60)
        pts = np.stack([x, 200 + 60 * np.sin(x / 40.0)], axis=1)
        shape = ac.recognize_shape(pts)
        self.assertEqual(shape.kind, "freehand")
        self.assertFalse(shape.closed)

    def test_tiny_scribble_is_ignored(self):
        rng = np.random.default_rng(1)
        pts = np.array([300.0, 200.0]) + rng.normal(0, 6, (30, 2))
        self.assertIsNone(ac.recognize_shape(pts))

    def test_too_few_points_is_ignored(self):
        self.assertIsNone(ac.recognize_shape([(0, 0), (100, 100), (200, 200)]))
        self.assertIsNone(ac.recognize_shape(None))


# --------------------------------------------------------------------------
# Board: catching, carrying, painting
# --------------------------------------------------------------------------

class ShapeBoardTests(unittest.TestCase):
    def setUp(self):
        self.board = ac.ShapeBoard(960, 540)
        self.shape = ac.recognize_shape(circle_stroke(), self.board.next_color())
        self.board.add(self.shape)
        self.palm = ac.palm_center(make_hand())

    def test_open_then_closed_hand_catches_and_moves_the_shape(self):
        start = self.shape.center.copy()
        self.board.update_hands([make_hand()], 0.0)
        self.assertFalse(self.shape.held)
        self.board.update_hands([make_hand(fist=True)], 0.1)
        self.assertTrue(self.shape.held)
        self.board.update_hands([make_hand((500.0, 300.0), fist=True)], 0.2)
        moved = self.shape.center - start
        self.assertAlmostEqual(float(moved[0]), 200.0, delta=1.0)
        self.assertAlmostEqual(float(moved[1]), 100.0, delta=1.0)

    def test_opening_the_hand_leaves_the_shape_in_place(self):
        self.board.update_hands([make_hand()], 0.0)
        self.board.update_hands([make_hand(fist=True)], 0.1)
        self.board.update_hands([make_hand((500.0, 300.0), fist=True)], 0.2)
        placed = self.shape.center.copy()
        self.board.update_hands([make_hand((500.0, 300.0))], 0.3)
        self.board.update_hands([make_hand((700.0, 100.0))], 0.7)
        self.assertFalse(self.shape.held)
        self.assertTrue(np.allclose(self.shape.center, placed))
        self.assertFalse(self.board.holding)

    def test_fist_without_reaching_does_not_catch(self):
        self.board.update_hands([make_hand(fist=True)], 0.0)
        self.assertFalse(self.shape.held)

    def test_fist_far_from_the_shape_does_not_catch(self):
        self.board.update_hands([make_hand((800.0, 450.0))], 0.0)
        self.board.update_hands([make_hand((800.0, 450.0), fist=True)], 0.1)
        self.assertFalse(self.shape.held)

    def test_no_catch_while_the_pen_is_down(self):
        self.board.update_hands([make_hand()], 0.0)
        self.board.update_hands([make_hand(fist=True)], 0.1, allow_grab=False)
        self.assertFalse(self.shape.held)

    def test_short_tracking_drop_keeps_the_grip(self):
        self.board.update_hands([make_hand()], 0.0)
        self.board.update_hands([make_hand(fist=True)], 0.1)
        self.board.update_hands([], 0.2)
        self.assertTrue(self.shape.held)
        self.board.update_hands([make_hand((320.0, 210.0), fist=True)], 0.3)
        self.assertTrue(self.shape.held)

    def test_shape_stays_inside_the_picture(self):
        self.board.update_hands([make_hand()], 0.0)
        self.board.update_hands([make_hand(fist=True)], 0.1)
        self.board.update_hands([make_hand((5000.0, 5000.0), fist=True)], 0.2)
        self.assertLessEqual(float(self.shape.center[0]), 959.0)
        self.assertLessEqual(float(self.shape.center[1]), 539.0)

    def test_two_shapes_only_one_is_caught(self):
        other = ac.recognize_shape(circle_stroke(center=(330, 220)), self.board.next_color())
        self.board.add(other)
        self.board.update_hands([make_hand()], 0.0)
        self.board.update_hands([make_hand(fist=True)], 0.1)
        self.assertEqual(sum(s.held for s in self.board.shapes), 1)

    def test_undo_and_clear(self):
        self.assertTrue(self.board.undo())
        self.assertFalse(self.board.undo())
        self.board.add(self.shape)
        self.board.clear()
        self.assertEqual(self.board.shapes, [])

    def test_render_paints_without_touching_the_input(self):
        frame = np.full((540, 960, 3), 40, np.uint8)
        before = frame.copy()
        out = self.board.render(frame)
        self.assertTrue(np.array_equal(frame, before))
        self.assertFalse(np.array_equal(out, before))
        x, y = (self.shape.points()[0]).astype(int)
        self.assertFalse(np.array_equal(out[y, x], before[y, x]))

    def test_render_with_nothing_to_paint_returns_the_same_image(self):
        empty = ac.ShapeBoard(960, 540)
        frame = np.zeros((540, 960, 3), np.uint8)
        self.assertIs(empty.render(frame), frame)

    def test_render_shows_the_stroke_being_drawn(self):
        empty = ac.ShapeBoard(960, 540)
        frame = np.zeros((540, 960, 3), np.uint8)
        out = empty.render(frame, stroke=[(100, 100), (200, 150), (300, 120)])
        self.assertTrue(out.any())


class HandMeasureTests(unittest.TestCase):
    def test_open_and_closed_hands(self):
        self.assertGreater(ac.finger_reach(make_hand()), ac.FIST_OPEN)
        self.assertLess(ac.finger_reach(make_hand(fist=True)), ac.FIST_CLOSE)

    def test_pointing_hand_is_not_a_fist(self):
        hand = make_hand(fist=True)
        hand[8] = (hand[5][0], hand[9][1] - 95)          # index finger stretched out
        self.assertGreater(ac.finger_reach(hand), ac.FIST_OPEN)


# --------------------------------------------------------------------------
# Dissolve effect
# --------------------------------------------------------------------------

class VanishStyleTests(unittest.TestCase):
    def setUp(self):
        self.h, self.w = 90, 160
        self.frame = np.full((self.h, self.w, 3), 200, np.uint8)
        self.bg = np.full((self.h, self.w, 3), 20, np.uint8)
        self.mask = np.zeros((self.h, self.w), np.float32)
        self.mask[10:80, 50:110] = 1.0
        self.vanisher = fx.Vanisher(self.w, self.h)

    def test_progress_zero_shows_person(self):
        for style in fx.Vanisher.STYLES:
            out = self.vanisher.render(style, self.frame, self.bg, self.mask, 0.0, 1.0)
            self.assertEqual(int(out[40, 80, 0]), 200, style)
            self.assertEqual(int(out[40, 5, 0]), 20, style)

    def test_progress_one_is_background_only(self):
        for style in fx.Vanisher.STYLES:
            out = self.vanisher.render(style, self.frame, self.bg, self.mask, 1.0, 1.0)
            self.assertTrue(np.array_equal(out, self.bg), style)

    def test_middle_progress_is_mixed(self):
        for style in fx.Vanisher.STYLES:
            seen_mixed = False
            for now in (0.1, 0.2, 0.3, 0.4):             # glitch changes with time
                out = self.vanisher.render(style, self.frame, self.bg, self.mask, 0.5, now)
                region = out[10:80, 50:110, 0]
                seen_mixed |= bool((region > 150).any() and (region < 60).any())
            self.assertTrue(seen_mixed, style)

    def test_empty_mask_returns_background(self):
        for style in fx.Vanisher.STYLES:
            out = self.vanisher.render(style, self.frame, self.bg, np.zeros_like(self.mask), 0.5, 1.0)
            self.assertTrue(np.array_equal(out, self.bg), style)

    def test_inputs_are_never_changed(self):
        frame, bg, mask = self.frame.copy(), self.bg.copy(), self.mask.copy()
        for style in fx.Vanisher.STYLES:
            self.vanisher.render(style, self.frame, self.bg, self.mask, 0.5, 1.0)
        self.assertTrue(np.array_equal(frame, self.frame) and np.array_equal(bg, self.bg)
                        and np.array_equal(mask, self.mask))

    def test_nothing_outside_the_person_changes(self):
        for style in fx.Vanisher.STYLES:
            out = self.vanisher.render(style, self.frame, self.bg, self.mask, 0.5, 1.0)
            self.assertTrue((out[:, :20] == 20).all() and (out[:, 140:] == 20).all(), style)

    def test_unknown_style_falls_back_to_dissolve(self):
        out = self.vanisher.render("nope", self.frame, self.bg, self.mask, 0.0, 1.0)
        self.assertEqual(int(out[40, 80, 0]), 200)


class ModelTests(unittest.TestCase):
    def test_bundled_models_load(self):
        self.assertEqual(nv.self_test(), 0)


if __name__ == "__main__":
    unittest.main()
