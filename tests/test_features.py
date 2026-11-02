"""Tests for the version 2 features of Nirni Livewebcam. Run with:
    python -m unittest discover -s tests

Copyright (c) 2026 Sidagam Nirni Naidu. All rights reserved.
"""

import math
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import air_canvas as ac  # noqa: E402
import effects as fx  # noqa: E402
import gestures as gs  # noqa: E402
import media  # noqa: E402
import nirni_livewebcam as nv  # noqa: E402
import studio as st  # noqa: E402
import tracking as tr  # noqa: E402


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def pose_hand(center=(300.0, 200.0), size=100.0, pose="open", index_tip=None):
    """A made-up hand with all 21 landmarks. pose: open, fist or point."""
    cx, cy = center
    hand = np.zeros((21, 2), np.float32)
    hand[0] = (cx, cy + size)
    knuckles = {5: (cx - 0.45 * size, cy + 0.05 * size), 9: (cx, cy),
                13: (cx + 0.40 * size, cy + 0.08 * size), 17: (cx + 0.75 * size, cy + 0.20 * size)}
    hand[1:5] = (cx - 0.6 * size, cy + 0.7 * size)
    for mcp, (pip, dip, tip) in {5: (6, 7, 8), 9: (10, 11, 12), 13: (14, 15, 16), 17: (18, 19, 20)}.items():
        mx, my = knuckles[mcp]
        hand[mcp] = (mx, my)
        stretched = pose == "open" or (pose == "point" and mcp == 5)
        offsets = (-0.4, -0.65, -0.9) if stretched else (-0.2, 0.0, 0.25)
        hand[pip] = (mx, my + offsets[0] * size)
        hand[dip] = (mx, my + offsets[1] * size)
        hand[tip] = (mx, my + offsets[2] * size)
    if index_tip is not None:
        hand[8] = index_tip
    return hand


def circle_points(center=(300, 200), radius=80, turns=1.05, count=60):
    t = np.linspace(0, 2 * math.pi * turns, count)
    return np.stack([center[0] + radius * np.cos(t), center[1] + radius * np.sin(t)], axis=1)


def make_circle(board, center=(300, 200), radius=80):
    shape = ac.recognize_shape(circle_points(center, radius), board.next_color(),
                               board.thickness, board.rainbow)
    board.add(shape)
    return shape


def scene(w=320, h=180, seed=4):
    rng = np.random.default_rng(seed)
    base = cv2.resize(rng.integers(40, 120, (9, 16, 3), dtype=np.uint8), (w, h),
                      interpolation=cv2.INTER_CUBIC)
    return base


# --------------------------------------------------------------------------
# Hand poses
# --------------------------------------------------------------------------

class PoseTests(unittest.TestCase):
    def test_poses_are_told_apart(self):
        self.assertTrue(ac.is_pointing(pose_hand(pose="point")))
        self.assertFalse(ac.is_pointing(pose_hand(pose="open")))
        self.assertFalse(ac.is_pointing(pose_hand(pose="fist")))

    def test_poses_agree_with_fist_measure(self):
        self.assertGreater(ac.finger_reach(pose_hand(pose="open")), ac.FIST_OPEN)
        self.assertLess(ac.finger_reach(pose_hand(pose="fist")), ac.FIST_CLOSE)
        self.assertGreater(ac.finger_reach(pose_hand(pose="point")), ac.FIST_OPEN)

    def test_pointing_works_when_the_hand_is_rotated(self):
        hand = pose_hand(pose="point")
        angle = math.radians(70)
        rot = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
        self.assertTrue(ac.is_pointing((hand - hand[0]) @ rot.T + hand[0]))

    def test_duplicate_hands_are_merged(self):
        a = pose_hand((300, 200))
        twin = a + 3.0
        far = pose_hand((700, 200))
        self.assertEqual(len(ac.dedupe_hands([a, twin, far])), 2)

    def test_two_hands_side_by_side_are_kept(self):
        self.assertEqual(len(ac.dedupe_hands([pose_hand((300, 200)), pose_hand((520, 200))])), 2)


class PenDuplicateTests(unittest.TestCase):
    def test_one_hand_detected_twice_cannot_draw(self):
        pen = ac.AirPen()
        hand = pose_hand((300, 200), pose="point")
        twin = hand + np.float32(2.0)
        for k in range(30):
            pen.update([hand, twin], k * 0.03)
        self.assertFalse(pen.down)


# --------------------------------------------------------------------------
# Face gestures
# --------------------------------------------------------------------------

class WinkTests(unittest.TestCase):
    def feed(self, det, left, right, start=0.0, seconds=0.6, step=0.03):
        fired = []
        t = start
        while t < start + seconds:
            fired.append(det.update(left, right, t))
            t += step
        return fired

    def test_held_wink_fires_once(self):
        det = gs.WinkDetector()
        self.assertEqual(sum(self.feed(det, 0.9, 0.05)), 1)

    def test_either_eye_counts(self):
        det = gs.WinkDetector()
        self.assertEqual(sum(self.feed(det, 0.05, 0.9)), 1)

    def test_normal_blink_never_fires(self):
        self.assertEqual(sum(self.feed(gs.WinkDetector(), 0.95, 0.9, seconds=2.0)), 0)

    def test_quick_twitch_never_fires(self):
        det = gs.WinkDetector()
        self.assertEqual(sum(self.feed(det, 0.9, 0.05, seconds=0.12)), 0)

    def test_open_eyes_never_fire(self):
        self.assertEqual(sum(self.feed(gs.WinkDetector(), 0.05, 0.05, seconds=3.0)), 0)

    def test_needs_release_and_cooldown(self):
        det = gs.WinkDetector()
        self.assertEqual(sum(self.feed(det, 0.9, 0.05, seconds=2.0)), 1)     # one long wink = one event
        self.feed(det, 0.05, 0.05, start=2.0, seconds=0.2)
        self.assertEqual(sum(self.feed(det, 0.9, 0.05, start=2.2, seconds=0.6)), 1)

    def test_no_face_never_fires(self):
        self.assertEqual(sum(self.feed(gs.WinkDetector(), None, None, seconds=2.0)), 0)


class SmileTests(unittest.TestCase):
    def feed(self, det, smile, jaw=0.1, start=0.0, seconds=1.0, step=0.05):
        out, t = [], start
        while t < start + seconds:
            out.append(det.update(smile, jaw, t))
            t += step
        return out

    def test_wide_smile_held_fires_once(self):
        det = gs.SmileDetector()
        self.assertEqual(sum(self.feed(det, 0.8, seconds=3.0)), 1)

    def test_small_smile_never_fires(self):
        self.assertEqual(sum(self.feed(gs.SmileDetector(), 0.35, seconds=3.0)), 0)

    def test_talking_never_fires(self):
        self.assertEqual(sum(self.feed(gs.SmileDetector(), 0.8, jaw=0.7, seconds=3.0)), 0)

    def test_brief_smile_never_fires(self):
        self.assertEqual(sum(self.feed(gs.SmileDetector(), 0.8, seconds=0.3)), 0)

    def test_cooldown_between_triggers(self):
        det = gs.SmileDetector()
        self.assertEqual(sum(self.feed(det, 0.8, seconds=1.0)), 1)
        self.feed(det, 0.05, start=1.0, seconds=0.3)
        self.assertEqual(sum(self.feed(det, 0.8, start=1.3, seconds=2.0)), 0)      # inside the pause
        self.feed(det, 0.05, start=3.3, seconds=0.3)
        self.assertEqual(sum(self.feed(det, 0.8, start=7.5, seconds=1.0)), 1)


# --------------------------------------------------------------------------
# Dock and window
# --------------------------------------------------------------------------

class DockTests(unittest.TestCase):
    def make(self):
        dock = gs.Dock(dwell=0.5)
        dock.layout(960, 540, [("blur", "BLUR", None), ("snap", "SNAP", None)],
                    [("c0", "SKY", (255, 200, 60)), ("undo", "UNDO", None)])
        return dock

    def center(self, dock, key):
        x0, y0, x1, y1 = next(b for b in dock.buttons if b.key == key).rect
        return ((x0 + x1) / 2, (y0 + y1) / 2)

    def test_buttons_are_inside_the_picture(self):
        dock = self.make()
        for b in dock.buttons:
            x0, y0, x1, y1 = b.rect
            self.assertTrue(0 <= x0 < x1 <= 960 and 0 <= y0 < y1 <= 540)
        self.assertEqual({b.side for b in dock.buttons}, {"L", "R"})

    def test_pointing_and_waiting_presses_once(self):
        dock = self.make()
        hand = pose_hand(pose="point", index_tip=self.center(dock, "undo"))
        keys = [dock.update([hand], t) for t in np.arange(0, 2.0, 0.05)]
        self.assertEqual([k for k in keys if k], ["undo"])

    def test_leaving_and_coming_back_presses_again(self):
        dock = self.make()
        on = pose_hand(pose="point", index_tip=self.center(dock, "undo"))
        presses = []
        for t in np.arange(0, 1.0, 0.05):
            presses.append(dock.update([on], float(t)))
        for t in np.arange(1.0, 1.6, 0.05):
            presses.append(dock.update([], float(t)))
        for t in np.arange(1.6, 2.6, 0.05):
            presses.append(dock.update([on], float(t)))
        self.assertEqual([k for k in presses if k], ["undo", "undo"])

    def test_open_hand_does_not_press(self):
        dock = self.make()
        hand = pose_hand(pose="open", index_tip=self.center(dock, "undo"))
        self.assertFalse(any(dock.update([hand], t) for t in np.arange(0, 2.0, 0.05)))

    def test_short_dropout_keeps_progress(self):
        dock = self.make()
        on = pose_hand(pose="point", index_tip=self.center(dock, "undo"))
        for t in np.arange(0, 0.3, 0.05):
            dock.update([on], float(t))
        dock.update([], 0.32)
        self.assertEqual(dock.update([on], 0.4) or dock.update([on], 0.55), "undo")

    def test_inactive_dock_does_nothing(self):
        dock = self.make()
        hand = pose_hand(pose="point", index_tip=self.center(dock, "undo"))
        self.assertFalse(any(dock.update([hand], t, active=False) for t in np.arange(0, 2.0, 0.05)))

    def test_moving_between_buttons_restarts_the_wait(self):
        dock = self.make()
        a = pose_hand(pose="point", index_tip=self.center(dock, "undo"))
        b = pose_hand(pose="point", index_tip=self.center(dock, "c0"))
        for t in np.arange(0, 0.4, 0.05):
            dock.update([a], float(t))
        self.assertIsNone(dock.update([b], 0.45))
        self.assertLess(dock.progress, 0.3)


class WindowTrackerTests(unittest.TestCase):
    def hands(self, a=(100, 100), b=(400, 300)):
        return [pose_hand((a[0], a[1] + 54), 60, "point", index_tip=a),
                pose_hand((b[0], b[1] + 54), 60, "point", index_tip=b)]

    def test_rectangle_between_the_two_index_tips(self):
        tracker = gs.WindowTracker()
        rect = tracker.update(self.hands(), 0.0, bounds=(640, 480))
        self.assertEqual(rect, (100, 100, 400, 300))

    def test_corners_in_any_order(self):
        rect = gs.WindowTracker().update(self.hands((400, 300), (100, 100)), 0.0)
        self.assertEqual(rect, (100, 100, 400, 300))

    def test_one_hand_gives_nothing(self):
        self.assertIsNone(gs.WindowTracker().update(self.hands()[:1], 0.0))

    def test_open_hands_give_nothing(self):
        hands = [pose_hand((60, 300), 60, "open"), pose_hand((500, 300), 60, "open")]
        self.assertIsNone(gs.WindowTracker().update(hands, 0.0))

    def test_tiny_window_is_ignored(self):
        self.assertIsNone(gs.WindowTracker().update(self.hands((100, 100), (120, 120)), 0.0))

    def test_survives_a_short_dropout_then_ends(self):
        tracker = gs.WindowTracker()
        tracker.update(self.hands(), 0.0)
        self.assertIsNotNone(tracker.update([], 0.1))
        self.assertIsNone(tracker.update([], 0.5))

    def test_disabled_tracker_clears(self):
        tracker = gs.WindowTracker()
        tracker.update(self.hands(), 0.0)
        self.assertIsNone(tracker.update(self.hands(), 0.1, enabled=False))

    def test_movement_is_smoothed(self):
        tracker = gs.WindowTracker()
        tracker.update(self.hands(), 0.0)
        rect = tracker.update(self.hands((100, 100), (500, 300)), 0.03)
        self.assertTrue(400 < rect[2] < 500)

    def test_rectangle_is_clipped_to_the_picture(self):
        rect = gs.WindowTracker().update(self.hands((-50, 100), (700, 300)), 0.0, bounds=(640, 480))
        self.assertEqual((rect[0], rect[2]), (0, 640))


# --------------------------------------------------------------------------
# Shape board: colours, history, eraser, two hands
# --------------------------------------------------------------------------

class ColourTests(unittest.TestCase):
    def test_fixed_colour_is_used_for_every_shape(self):
        board = ac.ShapeBoard(960, 540)
        board.set_color(3)
        a, b = make_circle(board), make_circle(board, (600, 300))
        self.assertEqual(a.color, ac.COLORS[3][1])
        self.assertEqual(b.color, ac.COLORS[3][1])

    def test_auto_colour_changes_per_shape(self):
        board = ac.ShapeBoard(960, 540)
        colors = {make_circle(board, (300 + 5 * i, 200)).color for i in range(3)}
        self.assertEqual(len(colors), 3)

    def test_rainbow_shape_carries_vertex_colours(self):
        board = ac.ShapeBoard(960, 540)
        board.set_color(ac.RAINBOW_INDEX)
        shape = make_circle(board)
        self.assertTrue(shape.rainbow)
        self.assertEqual(len(shape.colors), len(shape.outline))
        self.assertGreater(len({tuple(c) for c in shape.colors}), 10)
        self.assertEqual(board.color_name(), "Rainbow")

    def test_bad_colour_index_is_refused(self):
        with self.assertRaises(ValueError):
            ac.ShapeBoard(10, 10).set_color(99)

    def test_thickness_is_clamped(self):
        board = ac.ShapeBoard(960, 540)
        for _ in range(30):
            board.change_thickness(+1)
        self.assertEqual(board.thickness, 10)
        for _ in range(30):
            board.change_thickness(-1)
        self.assertEqual(board.thickness, 2)

    def test_thickness_is_stored_on_the_shape(self):
        board = ac.ShapeBoard(960, 540)
        board.change_thickness(+3)
        self.assertEqual(make_circle(board).thickness, 6)


class HistoryTests(unittest.TestCase):
    def setUp(self):
        self.board = ac.ShapeBoard(960, 540)

    def test_undo_and_redo_of_adding(self):
        make_circle(self.board)
        make_circle(self.board, (600, 300))
        self.assertTrue(self.board.undo())
        self.assertEqual(len(self.board.shapes), 1)
        self.assertTrue(self.board.redo())
        self.assertEqual(len(self.board.shapes), 2)
        self.assertFalse(self.board.redo())

    def test_nothing_to_undo(self):
        self.assertFalse(self.board.undo())
        self.assertFalse(self.board.can_undo)

    def test_new_action_ends_the_redo_branch(self):
        make_circle(self.board)
        self.board.undo()
        make_circle(self.board, (600, 300))
        self.assertFalse(self.board.can_redo)

    def test_clear_can_be_undone(self):
        make_circle(self.board)
        make_circle(self.board, (600, 300))
        self.assertTrue(self.board.clear())
        self.assertEqual(self.board.shapes, [])
        self.board.undo()
        self.assertEqual(len(self.board.shapes), 2)

    def test_clearing_an_empty_board_adds_no_step(self):
        self.assertFalse(self.board.clear())
        self.assertFalse(self.board.can_undo)

    def test_moving_a_shape_can_be_undone(self):
        shape = make_circle(self.board)
        start = shape.center.copy()
        self.board.update_hands([pose_hand()], 0.0)
        self.board.update_hands([pose_hand(pose="fist")], 0.1)
        self.board.update_hands([pose_hand((500.0, 300.0), pose="fist")], 0.2)
        self.board.update_hands([pose_hand((500.0, 300.0))], 0.3)
        self.board.update_hands([pose_hand((500.0, 300.0))], 0.8)
        self.assertFalse(self.board.holding)
        moved = self.board.shapes[0].center.copy()
        self.assertGreater(float(np.linalg.norm(moved - start)), 100)
        self.board.undo()
        self.assertTrue(np.allclose(self.board.shapes[0].center, start, atol=1.0))
        self.board.redo()
        self.assertTrue(np.allclose(self.board.shapes[0].center, moved, atol=1.0))

    def test_catching_without_moving_adds_no_step(self):
        make_circle(self.board)
        self.board.update_hands([pose_hand()], 0.0)
        self.board.update_hands([pose_hand(pose="fist")], 0.1)
        self.board.update_hands([pose_hand()], 0.2)
        self.board.update_hands([], 0.8)
        self.board.undo()                       # undoes the circle, not a phantom move
        self.assertEqual(self.board.shapes, [])

    def test_undo_while_holding_is_safe(self):
        make_circle(self.board)
        self.board.update_hands([pose_hand()], 0.0)
        self.board.update_hands([pose_hand(pose="fist")], 0.1)
        self.board.update_hands([pose_hand((400.0, 250.0), pose="fist")], 0.2)
        self.assertTrue(self.board.undo())
        self.assertFalse(self.board.holding)

    def test_history_is_limited(self):
        for i in range(ac.HISTORY_LIMIT + 20):
            make_circle(self.board, (300 + i, 200))
        undone = 0
        while self.board.undo():
            undone += 1
        self.assertLessEqual(undone, ac.HISTORY_LIMIT)
        self.assertGreater(undone, 10)

    def test_shape_limit_keeps_the_newest(self):
        for i in range(ac.MAX_SHAPES + 5):
            make_circle(self.board, (100 + i * 10, 200))
        self.assertEqual(len(self.board.shapes), ac.MAX_SHAPES)


class EraserTests(unittest.TestCase):
    def setUp(self):
        self.board = ac.ShapeBoard(960, 540)
        self.circle = make_circle(self.board)

    def test_touching_the_outline_erases(self):
        self.assertEqual(self.board.erase_at([(380, 200)], 20, 0.0), 1)
        self.assertEqual(self.board.shapes, [])

    def test_inside_a_closed_shape_erases(self):
        self.assertEqual(self.board.erase_at([(300, 200)], 5, 0.0), 1)

    def test_far_away_erases_nothing(self):
        self.assertEqual(self.board.erase_at([(700, 450)], 20, 0.0), 0)
        self.assertEqual(len(self.board.shapes), 1)

    def test_open_line_is_erased_only_near_the_line(self):
        board = ac.ShapeBoard(960, 540)
        pts = np.stack([np.linspace(100, 400, 30), np.full(30, 300.0)], axis=1)
        board.add(ac.recognize_shape(pts))
        self.assertEqual(board.erase_at([(250, 360)], 20, 0.0), 0)
        self.assertEqual(board.erase_at([(250, 310)], 20, 0.0), 1)

    def test_one_sweep_is_one_undo_step(self):
        make_circle(self.board, (650, 300))
        make_circle(self.board, (300, 420), 50)
        self.board.erase_at([(380, 200)], 20, 0.0)
        self.board.erase_at([(730, 300)], 20, 0.1)
        self.board.erase_at([(300, 420)], 20, 0.2)
        self.assertEqual(self.board.shapes, [])
        self.board.update_hands([], 1.0)                   # sweep finished
        self.assertTrue(self.board.undo())
        self.assertEqual(len(self.board.shapes), 3)

    def test_undo_right_after_erasing_works(self):
        self.board.erase_at([(380, 200)], 20, 0.0)
        self.assertTrue(self.board.undo())
        self.assertEqual(len(self.board.shapes), 1)

    def test_a_shape_in_your_hand_is_not_erased(self):
        self.board.update_hands([pose_hand()], 0.0)
        self.board.update_hands([pose_hand(pose="fist")], 0.1)
        self.assertEqual(self.board.erase_at([(380, 200)], 20, 0.2), 0)


class TwoHandTests(unittest.TestCase):
    """Two closed hands resize and rotate the shape one of them holds."""

    def setUp(self):
        self.board = ac.ShapeBoard(960, 540)
        self.shape = make_circle(self.board, (300, 200), 80)
        self.t = 0.0
        self.step([pose_hand((300, 200))])
        self.step([pose_hand((300, 200), pose="fist")])
        self.assertTrue(self.shape.held)

    def step(self, hands, dt=0.05):
        self.t += dt
        self.board.update_hands(hands, self.t)

    def fists(self, left, right):
        return [pose_hand(left, pose="fist"), pose_hand(right, pose="fist")]

    def test_second_closed_hand_starts_the_resize(self):
        self.step(self.fists((300, 200), (600, 200)))
        self.assertTrue(self.board.manipulating)
        self.assertIsNotNone(self.shape.pinch)

    def test_second_hand_too_close_does_nothing(self):
        self.step(self.fists((300, 200), (330, 200)))
        self.assertFalse(self.board.manipulating)

    def test_moving_hands_apart_grows_the_shape(self):
        self.step(self.fists((300, 200), (500, 200)))
        self.step(self.fists((300, 200), (500, 200)))
        self.step(self.fists((300, 200), (700, 200)))
        self.assertAlmostEqual(self.shape.scale, 2.0, delta=0.05)

    def test_moving_hands_together_shrinks_it(self):
        self.step(self.fists((300, 200), (700, 200)))
        self.step(self.fists((300, 200), (500, 200)))
        self.assertAlmostEqual(self.shape.scale, 0.5, delta=0.05)

    def test_turning_the_hands_rotates_it(self):
        self.step(self.fists((300, 200), (500, 200)))
        self.step(self.fists((300, 200), (300 + 200 * math.cos(math.radians(40)),
                                          200 + 200 * math.sin(math.radians(40)))))
        self.assertAlmostEqual(math.degrees(self.shape.angle), 40, delta=1.0)

    def test_scale_is_limited(self):
        self.step(self.fists((300, 200), (400, 200)))
        self.step(self.fists((300, 200), (900, 200)))
        self.assertLessEqual(self.shape.scale, ac.MAX_SCALE)
        self.step(self.fists((300, 200), (400, 200)))
        self.step(self.fists((300, 200), (303, 200)))
        self.assertGreaterEqual(self.shape.scale, ac.MIN_SCALE)

    def test_bigger_shape_has_bigger_points_and_reach(self):
        before = float(np.linalg.norm(self.shape.points() - self.shape.center, axis=1).max())
        self.step(self.fists((300, 200), (500, 200)))
        self.step(self.fists((300, 200), (700, 200)))
        after = float(np.linalg.norm(self.shape.points() - self.shape.center, axis=1).max())
        self.assertAlmostEqual(after / before, 2.0, delta=0.1)
        self.assertGreater(self.shape.reach_radius, 150)

    def test_no_jump_when_the_second_hand_opens(self):
        self.step(self.fists((300, 200), (500, 200)))
        self.step(self.fists((300, 200), (700, 200)))
        centre = self.shape.center.copy()
        scale = self.shape.scale
        self.step([pose_hand((300, 200), pose="fist"), pose_hand((700, 200))])     # second hand opens
        self.step([pose_hand((300, 200), pose="fist"), pose_hand((700, 200))], dt=0.4)
        self.assertIsNone(self.shape.pinch)
        self.assertTrue(self.shape.held)
        self.assertTrue(np.allclose(self.shape.center, centre, atol=1.0))
        self.assertEqual(self.shape.scale, scale)
        self.step([pose_hand((350, 250), pose="fist")])                          # one hand carries on
        self.assertTrue(np.allclose(self.shape.center, centre + 50, atol=1.5))

    def test_first_hand_leaving_hands_over_to_the_second(self):
        self.step(self.fists((300, 200), (500, 200)))
        self.step([pose_hand((500, 200), pose="fist")])
        self.assertIsNone(self.shape.pinch)
        self.assertTrue(self.shape.held)
        self.step([pose_hand((550, 230), pose="fist")])
        self.assertTrue(self.shape.held)

    def test_shape_follows_the_midpoint(self):
        self.step(self.fists((300, 200), (500, 200)))
        start = self.shape.center.copy()
        self.step(self.fists((350, 250), (550, 250)))
        self.assertTrue(np.allclose(self.shape.center, start + (50, 50), atol=1.5))

    def test_resize_is_one_undo_step(self):
        self.step(self.fists((300, 200), (500, 200)))
        self.step(self.fists((300, 200), (700, 200)))
        self.step([], dt=0.5)
        self.step([], dt=0.5)
        self.assertFalse(self.board.holding)
        self.assertGreater(self.board.shapes[0].scale, 1.5)
        self.board.undo()
        self.assertEqual(self.board.shapes[0].scale, 1.0)

    def test_undo_restores_the_rotation_too(self):
        self.step(self.fists((300, 200), (500, 200)))
        for degrees in range(15, 100, 15):
            r = math.radians(degrees)
            self.step(self.fists((300, 200), (300 + 200 * math.cos(r), 200 + 200 * math.sin(r))))
        self.step([], dt=0.5)
        self.step([], dt=0.5)
        self.assertNotEqual(self.board.shapes[0].angle, 0.0)
        self.board.undo()
        self.assertEqual(self.board.shapes[0].angle, 0.0)

    def test_second_hand_does_not_steal_another_shape(self):
        other = make_circle(self.board, (700, 300), 60)
        self.step([pose_hand((300, 200), pose="fist"), pose_hand((700, 300))])
        self.step(self.fists((300, 200), (700, 300)))
        self.assertTrue(self.shape.held)
        self.assertTrue(self.shape.pinch is not None or other.held)
        self.assertEqual(sum(s.held for s in self.board.shapes), 1)


class BoardPaintingTests(unittest.TestCase):
    def test_draw_works_in_place_and_returns_the_canvas(self):
        board = ac.ShapeBoard(320, 180)
        make_circle(board, (100, 90), 40)
        canvas = np.zeros((180, 320, 3), np.uint8)
        self.assertIs(board.draw(canvas), canvas)
        self.assertTrue(canvas.any())

    def test_far_corners_stay_untouched(self):
        board = ac.ShapeBoard(640, 360)
        make_circle(board, (100, 100), 30)
        canvas = np.zeros((360, 640, 3), np.uint8)
        board.draw(canvas)
        self.assertFalse(canvas[200:, 300:].any())

    def test_rainbow_stroke_shows_many_colours(self):
        board = ac.ShapeBoard(320, 180)
        canvas = np.zeros((180, 320, 3), np.uint8)
        board.draw(canvas, stroke=[(20 + 4 * i, 90 + 20 * math.sin(i / 5)) for i in range(60)],
                   stroke_rainbow=True)
        self.assertGreater(len({tuple(p) for p in canvas.reshape(-1, 3)[::7]}), 40)

    def test_scaled_shape_is_painted_bigger(self):
        board = ac.ShapeBoard(640, 360)
        shape = make_circle(board, (320, 180), 40)
        small = board.render(np.zeros((360, 640, 3), np.uint8))
        shape.scale = 2.0
        big = board.render(np.zeros((360, 640, 3), np.uint8))
        self.assertGreater(int((big.sum(axis=2) > 0).sum()), int((small.sum(axis=2) > 0).sum()) * 1.5)

    def test_shapes_off_the_edge_do_not_crash(self):
        board = ac.ShapeBoard(320, 180)
        shape = make_circle(board, (100, 90), 40)
        shape.center = np.array([-500.0, 900.0], np.float32)
        board.draw(np.zeros((180, 320, 3), np.uint8))

    def test_rescale_keeps_shapes_in_place(self):
        board = ac.ShapeBoard(320, 180)
        shape = make_circle(board, (160, 90), 40)
        board.rescale(640, 360)
        self.assertTrue(np.allclose(board.shapes[0].center, (320, 180), atol=1))
        board.undo()
        board.redo()
        self.assertTrue(np.allclose(board.shapes[0].center, (320, 180), atol=1))


# --------------------------------------------------------------------------
# Effects
# --------------------------------------------------------------------------

class BackgroundTests(unittest.TestCase):
    def test_gain_follows_a_brighter_camera(self):
        bg = fx.BackgroundModel()
        base = scene()
        bg.set(base)
        brighter = np.clip(base.astype(np.float32) * 1.25, 0, 255).astype(np.uint8)
        for _ in range(40):
            bg.adapt(brighter)
        self.assertAlmostEqual(float(bg.gain.mean()), 1.25, delta=0.04)
        self.assertGreater(float(bg.current().mean()), float(base.mean()) * 1.18)

    def test_gain_uses_only_pixels_without_the_person(self):
        bg = fx.BackgroundModel()
        base = scene()
        bg.set(base)
        frame = base.copy()
        frame[:, 100:240] = 255                        # a big bright person
        mask = np.zeros(base.shape[:2], np.float32)
        mask[:, 100:240] = 1.0
        for _ in range(40):
            bg.adapt(frame, mask)
        self.assertAlmostEqual(float(bg.gain.mean()), 1.0, delta=0.03)

    def test_a_person_filling_most_of_the_picture_cannot_drag_the_gain(self):
        bg = fx.BackgroundModel()
        base = scene()
        bg.set(base)
        frame = np.full_like(base, 250)
        for _ in range(40):
            bg.adapt(frame)
        self.assertAlmostEqual(float(bg.gain.mean()), 1.0, delta=0.03)

    def test_gain_is_limited(self):
        bg = fx.BackgroundModel()
        base = np.full((90, 160, 3), 100, np.uint8)
        bg.set(base)
        for _ in range(80):
            bg.adapt(np.full_like(base, 160), mask=np.zeros(base.shape[:2], np.float32))
        self.assertLessEqual(float(bg.gain.max()), fx.BackgroundModel.GAIN_MAX + 1e-3)

    def test_unchanged_camera_keeps_the_original(self):
        bg = fx.BackgroundModel()
        base = scene()
        bg.set(base)
        bg.adapt(base)
        self.assertTrue(np.array_equal(bg.current(), base))

    def test_current_before_set_fails_clearly(self):
        with self.assertRaises(RuntimeError):
            fx.BackgroundModel().current()


class MaskerTests(unittest.TestCase):
    def setUp(self):
        self.bg = fx.BackgroundModel()
        self.base = scene(320, 180)
        self.bg.set(self.base)
        self.masker = fx.PersonMasker(self.bg)

    def test_no_background_and_no_model_gives_none(self):
        self.assertIsNone(fx.PersonMasker(fx.BackgroundModel()).compute(self.base, 0))
        self.assertFalse(fx.PersonMasker(fx.BackgroundModel()).available)

    def test_empty_scene_has_no_person(self):
        mask = self.masker.compute(self.base.copy(), 0)
        self.assertLess(float(mask.max()), 0.05)

    def test_person_is_found_where_the_picture_changed(self):
        frame = self.base.copy()
        frame[40:150, 100:200] = (30, 200, 240)
        mask = self.masker.compute(frame, 0)
        self.assertGreater(float(mask[90, 150]), 0.9)
        self.assertLess(float(mask[10, 10]), 0.05)
        self.assertEqual(self.masker.source, "background difference")

    def test_sensor_noise_is_not_a_person(self):
        rng = np.random.default_rng(1)
        frame = np.clip(self.base.astype(np.int16) + rng.integers(-4, 5, self.base.shape), 0, 255).astype(np.uint8)
        self.assertLess(float(self.masker.compute(frame, 0).max()), 0.05)

    def test_tiny_specks_are_dropped(self):
        frame = self.base.copy()
        frame[100:104, 100:104] = 255
        self.assertLess(float(self.masker.compute(frame, 0).max()), 0.05)

    def test_mask_has_the_frame_size(self):
        self.assertEqual(self.masker.compute(self.base, 0).shape, self.base.shape[:2])


class WindowEffectTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(3)
        self.canvas = rng.integers(0, 255, (180, 320, 3), dtype=np.uint8)
        self.rect = (60, 40, 220, 140)

    def inner(self, canvas):
        return canvas[48:132, 68:212]

    def test_blur_smooths_the_inside_only(self):
        before = self.canvas.copy()
        self.assertTrue(fx.apply_window(self.canvas, self.rect, "blur"))
        self.assertLess(float(self.inner(self.canvas).std()), float(self.inner(before).std()) * 0.5)
        self.assertTrue(np.array_equal(self.canvas[:30], before[:30]))
        self.assertTrue(np.array_equal(self.canvas[:, 250:], before[:, 250:]))

    def test_pixelate_makes_blocks(self):
        fx.apply_window(self.canvas, self.rect, "pixelate")
        inside = self.inner(self.canvas)
        same_as_neighbour = (inside[:, 1:] == inside[:, :-1]).all(axis=2).mean()
        self.assertGreater(same_as_neighbour, 0.7)

    def test_magnify_shows_a_zoomed_centre(self):
        canvas = np.zeros((180, 320, 3), np.uint8)
        canvas[85:95, 135:145] = 255                         # 10 x 10 white square in the middle of the box
        fx.apply_window(canvas, self.rect, "magnify", zoom=2.0)
        white = int((canvas[48:132, 68:212].min(axis=2) > 200).sum())
        self.assertGreater(white, 300)                       # about 20 x 20 now
        self.assertLess(white, 600)

    def test_magnify_near_the_edge_stays_inside_the_picture(self):
        self.assertTrue(fx.apply_window(self.canvas, (0, 0, 120, 90), "magnify"))
        self.assertTrue(fx.apply_window(self.canvas, (200, 100, 320, 180), "magnify"))

    def test_border_is_drawn(self):
        before = self.canvas.copy()
        fx.apply_window(self.canvas, self.rect, "blur")
        self.assertFalse(np.array_equal(self.canvas[40, 100], before[40, 100]))

    def test_tiny_or_outside_rectangles_do_nothing(self):
        before = self.canvas.copy()
        self.assertFalse(fx.apply_window(self.canvas, (10, 10, 14, 14), "blur"))
        self.assertFalse(fx.apply_window(self.canvas, (400, 300, 500, 400), "blur"))
        self.assertTrue(np.array_equal(before, self.canvas))

    def test_unknown_kind_is_refused(self):
        with self.assertRaises(ValueError):
            fx.apply_window(self.canvas, self.rect, "sparkle")


class GhostTests(unittest.TestCase):
    def setUp(self):
        self.frame = scene(320, 180)
        self.frame[40:150, 100:200] = (30, 200, 240)
        self.mask = np.zeros((180, 320), np.float32)
        self.mask[40:150, 100:200] = 1.0
        self.empty = np.zeros((180, 320), np.float32)

    def test_cutout_needs_a_person(self):
        self.assertIsNone(fx.make_cutout(self.frame, self.empty, 0.0))
        cut = fx.make_cutout(self.frame, self.mask, 0.0)
        self.assertIsNotNone(cut)
        self.assertEqual(cut.alpha.shape, cut.crop.shape[:2])

    def test_tiny_blobs_are_not_a_person(self):
        small = np.zeros((180, 320), np.float32)
        small[10:13, 10:13] = 1.0
        self.assertIsNone(fx.make_cutout(self.frame, small, 0.0))

    def test_clone_stays_where_it_was_made(self):
        ghost = fx.GhostEngine()
        ghost.set_mode("clones")
        self.assertTrue(ghost.add_clone(self.frame, self.mask, 0.0))
        canvas = scene(320, 180, seed=9)
        before = canvas.copy()
        ghost.render(canvas, 0.0)
        changed = (canvas != before).any(axis=2)
        ys, xs = np.nonzero(changed)
        self.assertTrue(28 <= ys.min() and ys.max() <= 165 and 85 <= xs.min() and xs.max() <= 215)

    def test_live_person_stands_in_front_of_the_clone(self):
        ghost = fx.GhostEngine()
        ghost.set_mode("clones")
        ghost.add_clone(self.frame, self.mask, 0.0)
        canvas = scene(320, 180, seed=9)
        before = canvas.copy()
        ghost.render(canvas, 0.0, live_mask=self.mask)
        self.assertTrue(np.array_equal(canvas[60:130, 120:180], before[60:130, 120:180]))

    def test_no_person_no_clone(self):
        ghost = fx.GhostEngine()
        self.assertFalse(ghost.add_clone(self.frame, self.empty, 0.0))
        self.assertFalse(ghost.add_clone(self.frame, None, 0.0))

    def test_only_the_newest_clones_are_kept(self):
        ghost = fx.GhostEngine(max_clones=2)
        for i in range(5):
            ghost.add_clone(self.frame, self.mask, i)
        self.assertEqual(len(ghost.clones), 2)
        self.assertEqual([c.t for c in ghost.clones], [3, 4])

    def test_trail_shows_echoes_of_earlier_moments(self):
        ghost = fx.GhostEngine()
        ghost.set_mode("trail")
        for i in range(10):
            ghost.feed(self.frame, self.mask, i * 0.1)
        canvas = scene(320, 180, seed=9)
        before = canvas.copy()
        ghost.render(canvas, 0.95)
        self.assertFalse(np.array_equal(canvas, before))

    def test_trail_history_does_not_grow_forever(self):
        ghost = fx.GhostEngine()
        ghost.set_mode("trail")
        for i in range(300):
            ghost.feed(self.frame, self.mask, i * 0.05)
        self.assertLess(len(ghost.history), 30)

    def test_mode_cycle_and_cleanup(self):
        ghost = fx.GhostEngine()
        self.assertEqual([ghost.cycle() for _ in range(3)], ["trail", "clones", "off"])
        ghost.set_mode("clones")
        ghost.add_clone(self.frame, self.mask, 0.0)
        ghost.set_mode("off")
        self.assertEqual(ghost.clones, [])
        with self.assertRaises(ValueError):
            ghost.set_mode("zombie")

    def test_off_mode_paints_nothing(self):
        ghost = fx.GhostEngine()
        canvas = scene(320, 180)
        before = canvas.copy()
        ghost.feed(self.frame, self.mask, 0.0)
        ghost.render(canvas, 0.0)
        self.assertTrue(np.array_equal(canvas, before))


class ParticleTests(unittest.TestCase):
    def test_confetti_spawns_and_falls(self):
        ps = fx.ParticleSystem(640, 360, seed=1)
        self.assertGreater(ps.confetti(320, 300), 100)
        self.assertTrue(ps.active)
        ps.update(0.0)
        top = float(ps.pos[ps.life > 0, 1].mean())
        for k in range(1, 40):
            ps.update(k * 0.033)
        self.assertNotEqual(float(ps.pos[ps.life > 0, 1].mean()), top)

    def test_confetti_is_drawn(self):
        ps = fx.ParticleSystem(640, 360, seed=1)
        ps.confetti(320, 300)
        for k in range(10):
            ps.update(k * 0.033)
        canvas = np.zeros((360, 640, 3), np.uint8)
        ps.render(canvas)
        self.assertTrue(canvas.any())

    def test_everything_ends(self):
        ps = fx.ParticleSystem(640, 360, seed=1)
        ps.confetti(320, 300)
        for k in range(400):
            ps.update(k * 0.033)
        self.assertFalse(ps.active)

    def test_pool_is_limited(self):
        ps = fx.ParticleSystem(640, 360, capacity=50, seed=1)
        for _ in range(5):
            ps.confetti(320, 300)
        self.assertLessEqual(int((ps.life > 0).sum()), 50)

    def test_puff_is_drawn_and_fades(self):
        ps = fx.ParticleSystem(640, 360, seed=1)
        ps.puff(320, 180)
        ps.update(0.0)
        ps.update(0.1)
        canvas = np.zeros((360, 640, 3), np.uint8)
        ps.render(canvas)
        self.assertTrue(canvas.any())
        for k in range(1, 60):
            ps.update(0.1 + k * 0.05)
        self.assertFalse(ps.active)

    def test_large_time_jump_is_harmless(self):
        ps = fx.ParticleSystem(640, 360, seed=1)
        ps.confetti(320, 300)
        ps.update(0.0)
        ps.update(500.0)
        self.assertTrue(np.isfinite(ps.pos).all())

    def test_rendering_nothing_costs_nothing(self):
        canvas = np.zeros((90, 160, 3), np.uint8)
        fx.ParticleSystem(160, 90).render(canvas)
        self.assertFalse(canvas.any())


class SegmenterFallbackTests(unittest.TestCase):
    def test_download_failure_is_a_clear_error_not_a_crash_of_the_program(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch("urllib.request.urlopen", side_effect=urllib.error.URLError("offline")):
                with self.assertRaises(RuntimeError):
                    fx.download_segmenter(Path(tmp) / "m.tflite")
            self.assertEqual(list(Path(tmp).iterdir()), [])          # no half-downloaded file

    def test_loader_falls_back_quietly_when_offline(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch("urllib.request.urlopen", side_effect=urllib.error.URLError("offline")):
                loader = fx.SegmenterLoader(Path(tmp) / "m.tflite")
                for _ in range(50):
                    if loader.status != "loading":
                        break
                    time.sleep(0.05)
            self.assertEqual(loader.status, "failed")
            self.assertIsNone(loader.segmenter)

    def test_disabled_loader_never_touches_the_network(self):
        with mock.patch("urllib.request.urlopen", side_effect=AssertionError("network used")):
            loader = fx.SegmenterLoader(Path("nowhere.tflite"), enabled=False)
            time.sleep(0.1)
        self.assertEqual(loader.status, "off")

    def test_damaged_model_file_is_removed_and_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "m.tflite"
            path.write_bytes(b"x" * 200_000)
            loader = fx.SegmenterLoader(path)
            for _ in range(100):
                if loader.status != "loading":
                    break
                time.sleep(0.05)
            self.assertEqual(loader.status, "failed")
            self.assertFalse(path.exists())

    def test_masker_survives_a_failing_segmenter(self):
        class Broken:
            def segment_for_video(self, *a):
                raise RuntimeError("boom")
        bg = fx.BackgroundModel()
        base = scene()
        bg.set(base)
        loader = mock.Mock(segmenter=Broken())
        masker = fx.PersonMasker(bg, loader)
        for i in range(8):
            mask = masker.compute(base, i)
            self.assertIsNotNone(mask)
        self.assertEqual(masker.source, "background difference")

    def test_mask_bbox(self):
        mask = np.zeros((100, 200), np.float32)
        mask[20:40, 50:90] = 1
        x0, y0, x1, y1 = fx.mask_bbox(mask, margin=4)
        self.assertTrue(x0 <= 50 and y0 <= 20 and x1 >= 90 and y1 >= 40)
        self.assertIsNone(fx.mask_bbox(np.zeros((10, 10), np.float32)))


# --------------------------------------------------------------------------
# Media
# --------------------------------------------------------------------------

class PacerTests(unittest.TestCase):
    def test_steady_stream_gives_the_right_count(self):
        pacer = media.FramePacer(30)
        total = sum(pacer.due(k / 60.0) for k in range(120))        # camera at 60 fps for 2 s
        self.assertAlmostEqual(total, 60, delta=2)

    def test_slow_stream_repeats_frames(self):
        pacer = media.FramePacer(30)
        total = sum(pacer.due(k / 10.0) for k in range(20))         # 10 fps for 2 s
        self.assertAlmostEqual(total, 60, delta=6)

    def test_long_stall_does_not_flood(self):
        pacer = media.FramePacer(30)
        pacer.due(0.0)
        self.assertLessEqual(pacer.due(60.0), 5)
        self.assertLessEqual(pacer.due(60.01), 5)


class RecorderTests(unittest.TestCase):
    def test_a_real_video_is_written_with_the_right_length(self):
        with tempfile.TemporaryDirectory() as tmp:
            rec = media.Recorder(Path(tmp), fps=20)
            path = rec.start((160, 90), 0.0)
            frame = np.zeros((90, 160, 3), np.uint8)
            for k in range(100):                                    # 50 camera fps for 2 s
                frame[:] = k
                rec.write(frame, k / 50.0)
            saved, frames = rec.stop()
            self.assertEqual(saved, path)
            self.assertTrue(path.exists() and path.stat().st_size > 0)
            self.assertAlmostEqual(frames, 40, delta=3)
            cap = cv2.VideoCapture(str(path))
            self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), frames)
            cap.release()

    def test_frames_are_copied_not_shared(self):
        with tempfile.TemporaryDirectory() as tmp:
            rec = media.Recorder(Path(tmp), fps=10)
            rec.start((64, 48), 0.0)
            frame = np.full((48, 64, 3), 7, np.uint8)
            rec.write(frame, 0.0)
            frame[:] = 200
            rec.stop()                                              # must not crash or write 200

    def test_a_picture_size_change_while_recording_is_handled(self):
        with tempfile.TemporaryDirectory() as tmp:
            rec = media.Recorder(Path(tmp), fps=20)
            path = rec.start((160, 90), 0.0)
            for k in range(20):
                size = (160, 90) if k < 10 else (320, 180)
                rec.write(np.full((size[1], size[0], 3), 90, np.uint8), k / 20.0)
            _, frames = rec.stop()
            cap = cv2.VideoCapture(str(path))
            self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), frames)
            self.assertGreaterEqual(frames, 18)
            self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 160)
            cap.release()

    def test_stop_without_start_is_harmless(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(media.Recorder(Path(tmp)).stop(), (None, 0))

    def test_snapshot_is_a_png_with_a_unique_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            frame = np.full((40, 60, 3), 99, np.uint8)
            a = media.save_snapshot(frame, Path(tmp) / "new" / "folder")
            b = media.save_snapshot(frame, Path(tmp) / "new" / "folder")
            self.assertNotEqual(a, b)
            self.assertEqual(a.suffix, ".png")
            self.assertTrue(np.array_equal(cv2.imread(str(a)), frame))

    def test_snapshot_into_a_file_path_fails_cleanly(self):
        with tempfile.TemporaryDirectory() as tmp:
            blocker = Path(tmp) / "x"
            blocker.write_text("a file, not a folder")
            with self.assertRaises((OSError, RuntimeError)):
                media.save_snapshot(np.zeros((10, 10, 3), np.uint8), blocker)


class FakeCapture:
    def __init__(self, frames, fail_after=None):
        self.frames = list(frames)
        self.fail_after = fail_after
        self.released = False
        self.count = 0

    def read(self):
        time.sleep(0.002)
        self.count += 1
        if self.fail_after is not None and self.count > self.fail_after:
            return False, None
        return True, self.frames[self.count % len(self.frames)].copy()

    def release(self):
        self.released = True


class CameraStreamTests(unittest.TestCase):
    def test_hands_out_newer_pictures_only(self):
        frames = [np.full((8, 8, 3), i, np.uint8) for i in range(5)]
        stream = media.CameraStream(FakeCapture(frames), mirror=False)
        first = stream.read(0, timeout=1.0)
        second = stream.read(first[1], timeout=1.0)
        stream.close()
        self.assertGreater(second[1], first[1])

    def test_mirroring(self):
        frame = np.zeros((4, 8, 3), np.uint8)
        frame[:, 0] = 255
        stream = media.CameraStream(FakeCapture([frame]), mirror=True)
        got = stream.read(0, timeout=1.0)[0]
        stream.close()
        self.assertEqual(int(got[0, 7, 0]), 255)

    def test_unplugged_camera_raises_a_clear_error(self):
        with mock.patch.object(media.CameraStream, "FAIL_SECONDS", 0.1):
            stream = media.CameraStream(FakeCapture([np.zeros((4, 4, 3), np.uint8)], fail_after=3))
            with self.assertRaises(RuntimeError) as ctx:
                last = 0
                for _ in range(500):
                    item = stream.read(last, timeout=0.2)
                    if item:
                        last = item[1]
            stream.close()
            self.assertIn("webcam", str(ctx.exception).lower())

    def test_close_releases_the_camera(self):
        capture = FakeCapture([np.zeros((4, 4, 3), np.uint8)])
        stream = media.CameraStream(capture)
        stream.close()
        self.assertTrue(capture.released)


class WorkerTests(unittest.TestCase):
    class Fake:
        def __init__(self, delay=0.0, fail=False):
            self.delay, self.fail, self.calls, self.closed = delay, fail, [], False

        def detect(self, frame, now_ms, frame_id=0):
            self.calls.append(frame_id)
            time.sleep(self.delay)
            if self.fail:
                raise ValueError("bad frame")
            return tr.Observation(frame_id=frame_id)

        def close(self):
            self.closed = True

    def test_synchronous_mode(self):
        worker = tr.DetectionWorker(self.Fake(), threaded=False)
        worker.submit(np.zeros((4, 4, 3), np.uint8), 5)
        self.assertEqual(worker.latest().frame_id, 5)

    def test_threaded_worker_skips_old_pictures(self):
        fake = self.Fake(delay=0.05)
        worker = tr.DetectionWorker(fake)
        for i in range(1, 30):
            worker.submit(np.zeros((4, 4, 3), np.uint8), i)
            time.sleep(0.005)
        time.sleep(0.3)
        worker.close()
        self.assertLess(len(fake.calls), 15)
        self.assertEqual(fake.calls[-1], 29)
        self.assertTrue(fake.closed)

    def test_submit_never_blocks_on_a_slow_detector(self):
        worker = tr.DetectionWorker(self.Fake(delay=0.5))
        started = time.perf_counter()
        for i in range(20):
            worker.submit(np.zeros((4, 4, 3), np.uint8), i)
        self.assertLess(time.perf_counter() - started, 0.1)
        worker.close()

    def test_a_few_bad_pictures_are_ignored_but_endless_failure_is_reported(self):
        worker = tr.DetectionWorker(self.Fake(fail=True), threaded=False)
        for i in range(10):
            worker.submit(np.zeros((4, 4, 3), np.uint8), i)
        self.assertIsNone(worker.latest())
        for i in range(40):
            worker.submit(np.zeros((4, 4, 3), np.uint8), i)
        with self.assertRaises(RuntimeError):
            worker.latest()


# --------------------------------------------------------------------------
# The studio: everything together
# --------------------------------------------------------------------------

W, H = 320, 180
NOSE_XY, FOREHEAD_XY, CHIN_XY = (160.0, 100.0), (160.0, 50.0), (160.0, 150.0)


class StudioCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.studio = st.Studio(W, H, hold=0.3, output_dir=self.tmp.name)
        self.base = scene(W, H)
        self.studio.background.set(self.base)
        self.studio.phase = "ready"
        self.t = 0.0
        self.id = 0

    def obs(self, hands=(), face=True, **extra):
        self.id += 1
        data = dict(hands=list(hands), face_found=face, frame_id=self.id, nose=NOSE_XY,
                    forehead=FOREHEAD_XY, chin=CHIN_XY, mouth=(160.0, 125.0), face_width=60.0)
        if not face:
            data.update(nose=None, forehead=None, chin=None, mouth=None, face_width=0.0)
        data.update(extra)
        return tr.Observation(**data)

    def run_for(self, seconds, hands=(), frame=None, step=0.05, **extra):
        out = None
        end = self.t + seconds
        while self.t < end:
            self.t += step
            out = self.studio.process(self.base.copy() if frame is None else frame,
                                      self.obs(hands, **extra), self.t)
        return out

    def touch(self, point, seconds=0.6):
        hand = pose_hand((point[0], point[1] + 30), 20, "point", index_tip=point)
        self.run_for(seconds, [hand])
        self.run_for(0.3, [])                         # finger leaves

    def key(self, ch):
        return self.studio.on_key(ch, self.t)


class TouchAndModeTests(StudioCase):
    def test_forehead_toggles_draw_tools(self):
        self.touch(FOREHEAD_XY)
        self.assertEqual(self.studio.mode, "draw")
        self.touch(FOREHEAD_XY)
        self.assertEqual(self.studio.mode, "normal")

    def test_chin_toggles_the_effect_window(self):
        self.touch(CHIN_XY)
        self.assertEqual(self.studio.mode, "window")
        self.touch(CHIN_XY)
        self.assertEqual(self.studio.mode, "normal")

    def test_entering_one_tool_mode_leaves_the_other(self):
        self.touch(FOREHEAD_XY)
        self.touch(CHIN_XY)
        self.assertEqual(self.studio.mode, "window")
        self.assertEqual(self.studio.dock.buttons[0].side, "L")
        self.assertFalse(any(b.side == "R" for b in self.studio.dock.buttons))

    def test_nose_touch_vanishes_then_returns(self):
        self.touch(NOSE_XY)
        self.assertEqual(self.studio.phase, "vanishing")
        self.run_for(st.VANISH_SECONDS + 0.5)
        self.assertEqual(self.studio.phase, "vanished")
        self.touch(NOSE_XY)
        self.assertEqual(self.studio.phase, "appearing")
        self.run_for(st.APPEAR_SECONDS + 0.5)
        self.assertEqual(self.studio.phase, "ready")

    def test_nose_touch_halfway_reverses_the_vanish(self):
        self.touch(NOSE_XY)
        self.run_for(1.0)
        self.assertEqual(self.studio.phase, "vanishing")
        self.touch(NOSE_XY)
        self.assertEqual(self.studio.phase, "appearing")
        self.assertGreater(self.studio._vanish_progress(self.t), 0.2)    # fades back from where it was

    def test_nose_touch_while_appearing_turns_round(self):
        self.studio.phase, self.studio.phase_started = "appearing", self.t
        self.studio.begin_vanish(self.t + 0.5)
        self.assertEqual(self.studio.phase, "vanishing")

    def test_keys_vanish_and_reappear(self):
        self.key("v")
        self.assertEqual(self.studio.phase, "vanishing")
        self.key("r")
        self.assertEqual(self.studio.phase, "appearing")

    def test_style_key_cycles_and_unknown_style_is_ignored(self):
        names = []
        for _ in range(5):
            self.key("t")
            names.append(self.studio.vanish_style)
        self.assertEqual(names[:4], ["pixel", "portal", "glitch", "dissolve"])
        self.assertEqual(st.Studio(W, H, style="nope", output_dir=self.tmp.name).vanish_style, "dissolve")

    def test_every_style_renders_through_the_studio(self):
        frame = self.base.copy()
        frame[40:150, 100:200] = (30, 200, 240)
        for style in fx.Vanisher.STYLES:
            studio = st.Studio(W, H, style=style, output_dir=self.tmp.name)
            studio.background.set(self.base)
            studio.phase = "vanishing"
            studio.phase_started = 0.0
            out = studio.process(frame, None, 1.2)
            self.assertEqual(out.shape, frame.shape, style)
            self.assertFalse(np.array_equal(out, frame), style)

    def test_touches_are_ignored_while_calibrating(self):
        self.studio.phase = "calibrating"
        self.touch(FOREHEAD_XY)
        self.assertEqual(self.studio.mode, "normal")


class KeyTests(StudioCase):
    def test_quit_and_fullscreen_are_reported(self):
        self.assertEqual(self.key("q"), "quit")
        self.assertEqual(self.key("\x1b"), "quit")
        self.assertEqual(self.key("f"), "fullscreen")

    def test_every_documented_key_works_without_crashing(self):
        for ch in "dewgnzycp[]b0123456789shtvrm \r":
            self.t += 0.1
            self.key(ch)
            self.studio.process(self.base.copy(), self.obs(), self.t + 0.01)
        self.studio.close()

    def test_colour_keys(self):
        self.key("2")
        self.assertEqual(self.studio.board.color_name(), "Mint")
        self.key("8")
        self.assertTrue(self.studio.board.rainbow)
        self.key("0")
        self.assertIsNone(self.studio.board.color_index)

    def test_size_keys(self):
        self.key("]")
        self.key("]")
        self.assertEqual(self.studio.board.thickness, 5)
        self.key("[")
        self.assertEqual(self.studio.board.thickness, 4)

    def test_eraser_key_enters_draw_mode(self):
        self.key("e")
        self.assertEqual((self.studio.mode, self.studio.tool), ("draw", "eraser"))
        self.key("e")
        self.assertEqual(self.studio.tool, "pen")

    def test_window_key_cycles_kinds_then_off(self):
        seen = []
        for _ in range(4):
            self.key("w")
            seen.append((self.studio.mode, self.studio.window_kind))
        self.assertEqual(seen, [("window", "blur"), ("window", "pixelate"), ("window", "magnify"),
                                ("normal", "magnify")])

    def test_help_toggle(self):
        self.key("h")
        self.assertTrue(self.studio.help)

    def test_face_gesture_toggle(self):
        self.key("s")
        self.assertFalse(self.studio.face_gestures)
        self.key("s")
        self.assertTrue(self.studio.face_gestures)

    def test_space_recalibrates_and_keeps_the_shapes(self):
        make_circle(self.studio.board)
        self.key(" ")
        self.assertEqual(self.studio.phase, "calibrating")
        out = self.studio.process(self.base.copy(), self.obs(face=False), self.t + 0.1)
        self.assertFalse(np.array_equal(out, self.base))              # shapes are still painted

    def test_caps_lock_changes_nothing(self):
        for upper, lower in ((ord("Q"), "q"), (ord("Z"), "z"), (ord("D"), "d"), (ord("1"), "1")):
            self.assertEqual(nv.key_to_char(upper), lower)
        self.assertEqual(nv.key_to_char(27), "\x1b")
        self.assertEqual(nv.key_to_char(13), "\r")
        self.assertIsNone(nv.key_to_char(-1))
        self.assertIsNone(nv.key_to_char(0x270000))                    # arrow-like codes


class DockActionTests(StudioCase):
    def press(self, key):
        self.studio._dock_action(key, self.t)

    def test_colour_buttons(self):
        self.studio.tool = "eraser"
        self.press("c5")
        self.assertEqual(self.studio.board.color_name(), "Red")
        self.assertEqual(self.studio.tool, "pen")
        self.press("rainbow")
        self.assertTrue(self.studio.board.rainbow)

    def test_undo_redo_buttons(self):
        make_circle(self.studio.board)
        self.press("undo")
        self.assertEqual(self.studio.board.shapes, [])
        self.press("redo")
        self.assertEqual(len(self.studio.board.shapes), 1)

    def test_effect_buttons_switch_to_the_window(self):
        self.press("pixelate")
        self.assertEqual((self.studio.mode, self.studio.window_kind), ("window", "pixelate"))

    def test_snap_and_record_buttons(self):
        self.press("snap")
        self.assertGreater(self.studio.snapshot_at, self.t)
        self.press("rec")
        self.assertTrue(self.studio.recorder.recording)
        self.studio.close()
        self.assertFalse(self.studio.recorder.recording)

    def test_pointing_at_a_button_in_the_picture_presses_it(self):
        self.touch(FOREHEAD_XY)                                       # draw tools on
        x0, y0, x1, y1 = next(b for b in self.studio.dock.buttons if b.key == "c5").rect
        hand = pose_hand(((x0 + x1) / 2, y1 + 25), 15, "point", index_tip=((x0 + x1) / 2, (y0 + y1) / 2))
        self.run_for(1.2, [hand])
        self.assertEqual(self.studio.board.color_name(), "Red")


class DrawingTests(StudioCase):
    def joined(self, p):
        x, y = p
        return [pose_hand((x - 40, y + 20), 20, "point", index_tip=(x - 1, y)),
                pose_hand((x + 40, y + 20), 20, "point", index_tip=(x + 1, y))]

    def draw_circle(self):
        pts = circle_points((160, 90), 40, turns=1.05, count=45)
        for p in pts:
            self.t += 0.03
            self.studio.process(self.base.copy(), self.obs(self.joined(p)), self.t)
        self.run_for(0.6, [])

    def test_drawing_a_circle_in_draw_mode(self):
        self.key("d")
        self.key("3")
        self.draw_circle()
        self.assertEqual([s.kind for s in self.studio.board.shapes], ["circle"])
        self.assertEqual(self.studio.board.shapes[0].color, ac.COLORS[2][1])

    def test_no_drawing_when_tools_are_off(self):
        self.draw_circle()
        self.assertEqual(self.studio.board.shapes, [])

    def test_no_drawing_with_the_eraser(self):
        self.key("e")
        self.draw_circle()
        self.assertEqual(self.studio.board.shapes, [])

    def test_wink_undoes_the_last_shape(self):
        self.key("d")
        self.draw_circle()
        self.assertEqual(len(self.studio.board.shapes), 1)
        self.run_for(1.0, [], blink_left=0.9, blink_right=0.05)
        self.assertEqual(self.studio.board.shapes, [])

    def test_wink_does_nothing_when_gestures_are_off(self):
        self.key("d")
        self.draw_circle()
        self.key("s")
        self.run_for(1.0, [], blink_left=0.9, blink_right=0.05)
        self.assertEqual(len(self.studio.board.shapes), 1)

    def test_eraser_removes_a_shape_the_finger_touches(self):
        make_circle(self.studio.board, (160, 90), 40)
        self.key("e")
        hand = pose_hand((200, 140), 20, "point", index_tip=(200, 90))
        self.run_for(0.4, [hand])
        self.assertEqual(self.studio.board.shapes, [])
        self.run_for(1.0, [])
        self.key("z")
        self.assertEqual(len(self.studio.board.shapes), 1)

    def test_eraser_ignores_an_open_hand(self):
        make_circle(self.studio.board, (160, 90), 40)
        self.key("e")
        self.run_for(0.4, [pose_hand((200, 90), 20, "open")])
        self.assertEqual(len(self.studio.board.shapes), 1)

    def test_tracker_going_quiet_lifts_the_pen(self):
        self.key("d")
        pts = circle_points((160, 90), 40, turns=0.6, count=30)
        for p in pts:
            self.t += 0.03
            self.studio.process(self.base.copy(), self.obs(self.joined(p)), self.t)
        self.assertTrue(self.studio.pen.down)
        last = self.studio._latest_obs
        for _ in range(30):                                          # same stale result again and again
            self.t += 0.05
            self.studio.process(self.base.copy(), last, self.t)
        self.assertFalse(self.studio.pen.down)


class FeatureTests(StudioCase):
    def person_frame(self):
        frame = self.base.copy()
        frame[40:150, 100:200] = (30, 200, 240)
        return frame

    def test_window_effect_is_applied_between_two_pointing_hands(self):
        self.key("w")
        hands = [pose_hand((40, 120), 15, "point", index_tip=(90, 50)),
                 pose_hand((280, 120), 15, "point", index_tip=(230, 130))]
        out = self.run_for(0.6, hands)
        self.assertIsNotNone(self.studio.window.rect)
        inside = out[60:120, 100:220].astype(float)
        self.assertLess(inside.std(), self.base[60:120, 100:220].astype(float).std() * 0.7)

    def test_window_goes_away_when_the_hands_do(self):
        self.key("w")
        hands = [pose_hand((40, 120), 15, "point", index_tip=(90, 50)),
                 pose_hand((280, 120), 15, "point", index_tip=(230, 130))]
        self.run_for(0.5, hands)
        out = self.run_for(1.0, [])
        self.assertIsNone(self.studio.window.rect)
        self.assertTrue(np.array_equal(out, self.base))

    def test_clone_is_made_and_stays(self):
        out = self.studio.process(self.person_frame(), self.obs(), 1.0)
        self.key("n")
        self.assertEqual(len(self.studio.ghost.clones), 1)
        self.assertEqual(self.studio.ghost.mode, "clones")
        later = self.run_for(1.5, frame=self.base.copy())                 # you walked away
        diff = (later != self.base).any(axis=2)
        self.assertGreater(int(diff[40:150, 100:200].sum()), 2000)         # the clone is still there

    def test_clone_without_anyone_says_so(self):
        self.studio.process(self.base.copy(), self.obs(), 1.0)
        self.key("n")
        self.assertEqual(self.studio.ghost.clones, [])
        self.assertIn("Step into", self.studio.toast)

    def test_ghost_trail_works_through_the_studio(self):
        self.key("g")
        self.assertEqual(self.studio.ghost.mode, "trail")
        frames = [self.person_frame(), self.base.copy()]
        for i in range(30):
            self.t += 0.05
            self.studio.process(frames[(i // 6) % 2], self.obs(), self.t)
        self.assertGreater(len(self.studio.ghost.history), 3)

    def test_smile_makes_confetti_that_is_in_the_output(self):
        self.run_for(1.0, [], smile=0.9, jaw_open=0.1)
        self.assertTrue(self.studio.particles.active)
        out = self.run_for(0.3, [], smile=0.9, jaw_open=0.1)
        self.assertFalse(np.array_equal(out, self.base))

    def test_talking_does_not_make_confetti(self):
        self.run_for(3.0, [], smile=0.9, jaw_open=0.8)
        self.assertFalse(self.studio.particles.active)

    def test_burst_key(self):
        self.key("b")
        self.assertTrue(self.studio.particles.active)

    def test_snapshot_key_saves_the_clean_picture(self):
        self.key("d")
        self.studio.process(self.base.copy(), self.obs(), self.t + 0.05)
        self.key("p")
        out = self.studio.process(self.base.copy(), self.obs(), self.t + 0.1)
        files = list(Path(self.tmp.name).glob("*.png"))
        self.assertEqual(len(files), 1)
        self.assertTrue(np.array_equal(cv2.imread(str(files[0])), out))

    def test_snap_button_waits_for_the_countdown(self):
        self.studio._dock_action("snap", self.t)
        self.run_for(1.0, [])
        self.assertEqual(list(Path(self.tmp.name).glob("*.png")), [])
        self.run_for(3.0, [])
        self.assertEqual(len(list(Path(self.tmp.name).glob("*.png"))), 1)

    def test_recording_through_the_studio(self):
        self.key("m")
        self.run_for(1.0, [])
        self.key("m")
        videos = list(Path(self.tmp.name).glob("nirni-*"))
        self.assertEqual(len(videos), 1)
        cap = cv2.VideoCapture(str(videos[0]))
        self.assertGreater(cap.get(cv2.CAP_PROP_FRAME_COUNT), 10)
        self.assertEqual((int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))), (W, H))
        cap.release()

    def test_unwritable_output_folder_is_a_message_not_a_crash(self):
        blocker = Path(self.tmp.name) / "file"
        blocker.write_text("x")
        studio = st.Studio(W, H, output_dir=blocker)
        studio.background.set(self.base)
        studio.phase = "ready"
        studio.on_key("p", 0.0)
        studio.process(self.base.copy(), None, 0.1)
        self.assertIn("Could not", studio.toast)
        studio.on_key("m", 0.2)
        self.assertFalse(studio.recorder.recording)

    def test_input_frame_and_background_are_never_painted_on(self):
        make_circle(self.studio.board, (160, 90), 40)
        self.key("w")
        self.key("b")
        frame = self.person_frame()
        before, bg_before = frame.copy(), self.studio.background.base.copy()
        self.studio.process(frame, self.obs(), 5.0)
        self.studio.phase = "vanished"
        self.studio.process(frame, self.obs(), 5.1)
        self.assertTrue(np.array_equal(frame, before))
        self.assertTrue(np.array_equal(self.studio.background.base, bg_before))

    def test_shapes_are_visible_while_vanished(self):
        make_circle(self.studio.board, (160, 90), 40)
        self.studio.phase = "vanished"
        out = self.studio.process(self.base.copy(), self.obs(), 1.0)
        self.assertFalse(np.array_equal(out, self.studio.background.current()))

    def test_picture_size_change_is_survived(self):
        make_circle(self.studio.board, (160, 90), 40)
        out = self.studio.process(cv2.resize(self.base, (640, 360)), self.obs(), 1.0)
        self.assertEqual(out.shape, (360, 640, 3))
        self.assertEqual((self.studio.board.width, self.studio.board.height), (640, 360))
        self.assertEqual(self.studio.background.base.shape, (360, 640, 3))


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.studio = st.Studio(W, H, output_dir=self.tmp.name)
        self.studio.phase_started = 0.0
        self.base = scene(W, H)
        self.id = 0

    def step(self, t, frame=None, face=False, hands=()):
        self.id += 1
        obs = tr.Observation(face_found=face, hands=list(hands), frame_id=self.id)
        self.studio.process(self.base.copy() if frame is None else frame, obs, t)

    def test_empty_still_scene_is_saved_after_the_countdown(self):
        for k in range(200):
            self.step(k * 0.05)
        self.assertEqual(self.studio.phase, "ready")
        self.assertTrue(np.array_equal(self.studio.background.base, self.base))

    def test_not_saved_too_early(self):
        for k in range(50):                         # 2.5 s
            self.step(k * 0.05)
        self.assertEqual(self.studio.phase, "calibrating")

    def test_person_in_the_picture_blocks_saving(self):
        for k in range(300):                        # 15 s with a face in view
            self.step(k * 0.05, face=True)
        self.assertEqual(self.studio.phase, "calibrating")
        self.assertIn("Step out", self.studio.cal_status)

    def test_a_hand_in_the_picture_blocks_saving(self):
        for k in range(300):
            self.step(k * 0.05, hands=[pose_hand((100, 100), 20)])
        self.assertEqual(self.studio.phase, "calibrating")

    def test_movement_blocks_saving(self):
        rng = np.random.default_rng(0)
        for k in range(300):
            self.step(k * 0.05, frame=scene(W, H, seed=int(rng.integers(0, 10**6))))
        self.assertEqual(self.studio.phase, "calibrating")
        self.assertEqual(self.studio.cal_status, "Hold still")

    def test_saving_starts_over_when_someone_walks_through(self):
        for k in range(70):
            self.step(k * 0.05)
        self.step(3.6, face=True)                   # someone appears right before the end
        for k in range(73, 90):
            self.step(k * 0.05)
        self.assertEqual(self.studio.phase, "calibrating")
        for k in range(90, 200):
            self.step(k * 0.05)
        self.assertEqual(self.studio.phase, "ready")

    def test_saved_background_has_no_person_in_it(self):
        person = self.base.copy()
        person[40:150, 100:200] = 255
        for k in range(60):                         # person leaves at 3 s
            self.step(k * 0.05, frame=person if k < 40 else None, face=k < 40)
        for k in range(60, 200):
            self.step(k * 0.05)
        self.assertEqual(self.studio.phase, "ready")
        self.assertTrue(np.array_equal(self.studio.background.base, self.base))

    def test_enter_saves_right_away(self):
        self.step(0.1, face=True)
        self.studio.on_key("\r", 0.2)
        self.step(0.3, face=True)
        self.assertEqual(self.studio.phase, "ready")
        self.assertIn("press SPACE", self.studio.toast)

    def test_it_never_waits_forever(self):
        for k in range(int(st.CALIBRATION_TIMEOUT / 0.5) + 5):
            self.step(k * 0.5, face=True)
        self.assertEqual(self.studio.phase, "ready")


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------

class CommandLineTests(unittest.TestCase):
    def test_defaults(self):
        args = nv.parse_args([])
        self.assertEqual((args.width, args.height), (1280, 720))
        self.assertEqual(args.style, "dissolve")
        self.assertFalse(args.no_face_gestures)

    def test_fast_preset(self):
        args = nv.parse_args(["--fast"])
        self.assertEqual((args.width, args.height, args.detect_width), (640, 360, 320))

    def test_fast_keeps_an_explicit_size(self):
        args = nv.parse_args(["--fast", "--width", "800", "--height", "450"])
        self.assertEqual((args.width, args.height), (800, 450))

    def test_bad_values_are_refused(self):
        for bad in (["--width", "5"], ["--height", "99999"], ["--fps", "0"], ["--fps", "nan"],
                    ["--hold", "0"], ["--hold", "99"], ["--touch-distance", "-1"],
                    ["--chin-distance", "9"], ["--pen-join", "0"], ["--style", "fire"]):
            with self.assertRaises(SystemExit, msg=str(bad)), mock.patch("sys.stderr"):
                nv.parse_args(bad)

    def test_smile_level_is_clamped(self):
        self.assertEqual(nv.parse_args(["--smile-level", "5"]).smile_level, 0.95)
        self.assertEqual(nv.parse_args(["--smile-level", "-5"]).smile_level, 0.1)

    def test_fast_keeps_an_explicit_size_given_with_equals(self):
        args = nv.parse_args(["--fast", "--width=800"])
        self.assertEqual((args.width, args.height), (800, 360))

    def test_output_dir_option(self):
        self.assertEqual(nv.parse_args(["--output-dir", "x"]).output_dir, Path("x"))


class RealModelTests(unittest.TestCase):
    def test_the_real_detector_runs_on_a_blank_picture(self):
        detector = tr.create_detector()
        try:
            obs = detector.detect(np.zeros((360, 640, 3), np.uint8), 100, frame_id=3)
        finally:
            detector.close()
        self.assertEqual(obs.frame_id, 3)
        self.assertFalse(obs.face_found)
        self.assertEqual(obs.hands, [])

    def test_detector_without_blendshapes_loads(self):
        detector = tr.create_detector(face_gestures=False)
        detector.close()


if __name__ == "__main__":
    unittest.main()
