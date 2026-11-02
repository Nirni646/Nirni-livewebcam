"""Visual effects for Nirni Livewebcam.

Copyright (c) 2026 Sidagam Nirni Naidu. All rights reserved.
Use of this file is governed by the LICENSE file in this folder.

  BackgroundModel  the saved empty background, kept matched to the camera's
                   exposure so the vanish never shows a seam
  PersonMasker     where you are in the picture: the MediaPipe body-outline
                   model when it is available, otherwise a difference against
                   the saved background (works with no internet at all)
  Vanisher         four ways to disappear: dissolve, pixel, portal, glitch
  GhostEngine      ghost trail and shadow clones
  apply_window     blur / pixelate / magnifier inside a rectangle
  ParticleSystem   confetti and smoke puffs

Every effect works only on the part of the picture it changes (a "region of
interest"), which is what keeps the frame rate high.
"""

from __future__ import annotations

import math
import os
import threading
import urllib.error
import urllib.request
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

SEGMENTER_NAME = "selfie_segmenter_landscape.tflite"
SEGMENTER_URL = (
    "https://storage.googleapis.com/mediapipe-models/image_segmenter/"
    "selfie_segmenter_landscape/float16/latest/selfie_segmenter_landscape.tflite"
)


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------

def smoothstep(edge0: float, edge1: float, values: np.ndarray) -> np.ndarray:
    values = np.clip((values - edge0) / max(edge1 - edge0, 1e-6), 0.0, 1.0)
    return values * values * (3.0 - 2.0 * values)


def ease_in_out_cubic(value: float) -> float:
    if value < 0.5:
        return 4.0 * value * value * value
    return 1.0 - ((-2.0 * value + 2.0) ** 3) / 2.0


def blend(top: np.ndarray, bottom: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """top over bottom, alpha is float32 (h, w) in 0..1. Fast SIMD path."""
    alpha = np.ascontiguousarray(alpha, dtype=np.float32)
    return cv2.blendLinear(np.ascontiguousarray(top), np.ascontiguousarray(bottom),
                           alpha, 1.0 - alpha)


def alpha_composite(person: np.ndarray, background: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    return blend(person, background, alpha)


def mask_bbox(mask: np.ndarray, margin: int = 24, threshold: float = 0.01, step: int = 4):
    """Box (x0, y0, x1, y1) around everything above the threshold, or None.
    Looks at every few pixels only, which is plenty and much faster."""
    small = mask[::step, ::step] > threshold
    rows = np.flatnonzero(small.any(axis=1))
    if rows.size == 0:
        return None
    cols = np.flatnonzero(small.any(axis=0))
    height, width = mask.shape[:2]
    return (max(0, int(cols[0]) * step - margin), max(0, int(rows[0]) * step - margin),
            min(width, (int(cols[-1]) + 1) * step + margin),
            min(height, (int(rows[-1]) + 1) * step + margin))


# --------------------------------------------------------------------------
# Background and person mask
# --------------------------------------------------------------------------

class BackgroundModel:
    """The saved empty background, with a gain that follows the camera's
    auto-exposure and white balance so the vanish never shows a silhouette."""

    GAIN_MIN, GAIN_MAX = 0.6, 1.6

    def __init__(self):
        self.base: np.ndarray | None = None
        self.gain = np.ones(3, np.float32)
        self._applied = np.ones(3, np.float32)
        self._current: np.ndarray | None = None
        self._small: np.ndarray | None = None

    @property
    def ready(self) -> bool:
        return self.base is not None

    def set(self, image: np.ndarray) -> None:
        self.base = image.copy()
        self.gain = np.ones(3, np.float32)
        self._applied = np.ones(3, np.float32)
        self._current = self.base
        height, width = image.shape[:2]
        self._small = cv2.resize(image, (max(8, width // 8), max(8, height // 8)),
                                 interpolation=cv2.INTER_AREA).astype(np.float32)

    def current(self) -> np.ndarray:
        """The background as it should look right now. Do not edit the result."""
        if self.base is None:
            raise RuntimeError("no background saved yet")
        if self._current is None or float(np.abs(self.gain - self._applied).max()) > 0.004:
            table = np.clip(np.arange(256, dtype=np.float32)[:, None] * self.gain[None, :],
                            0, 255).astype(np.uint8).reshape(256, 1, 3)
            self._current = cv2.LUT(self.base, table)
            self._applied = self.gain.copy()
        return self._current

    def adapt(self, frame: np.ndarray, mask: np.ndarray | None = None, rate: float = 0.15) -> bool:
        """Nudge the gain towards what the camera sees now. With a person mask
        only the pixels without you are used; without one, the median of the
        whole picture is, which is steady while you cover less than half of it."""
        if self._small is None:
            return False
        small_h, small_w = self._small.shape[:2]
        live = cv2.resize(frame, (small_w, small_h), interpolation=cv2.INTER_AREA).astype(np.float32)
        ratio = (live + 8.0) / (self._small + 8.0)
        if mask is not None:
            small_mask = cv2.resize(mask, (small_w, small_h), interpolation=cv2.INTER_AREA)
            keep = cv2.erode((small_mask < 0.03).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
            if keep.mean() < 0.15:
                return False
            estimate = np.median(ratio[keep], axis=0)
        else:
            # No mask: trust only pixels that already agree with the current gain
            # (the empty background), so a person filling the picture cannot
            # drag the gain away.
            inliers = np.all(np.abs(ratio / self.gain - 1.0) < 0.18, axis=2)
            if inliers.mean() >= 0.25:
                estimate = np.median(ratio[inliers], axis=0)
            else:
                # A big jump (lights switched on). Believe it only when the top
                # and side edges of the picture, where you rarely are, all agree.
                estimate = self._edge_consensus(ratio)
                if estimate is None:
                    return False
        estimate = np.clip(estimate, self.GAIN_MIN, self.GAIN_MAX).astype(np.float32)
        self.gain = self.gain + rate * (estimate - self.gain)
        return True


    @staticmethod
    def _edge_consensus(ratio: np.ndarray):
        h, w = ratio.shape[:2]
        ring = np.concatenate([ratio[:max(1, h // 8)].reshape(-1, 3),
                               ratio[:, :max(1, w // 10)].reshape(-1, 3),
                               ratio[:, -max(1, w // 10):].reshape(-1, 3)])
        middle = np.median(ring, axis=0)
        agree = np.all(np.abs(ring / middle - 1.0) < 0.1, axis=1)
        if agree.mean() < 0.6:
            return None
        return np.median(ring[agree], axis=0)


def download_segmenter(model_path: Path, timeout: float = 25.0) -> None:
    """Fetch the body-outline model (once). Raises on any failure."""
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    if model_path.exists() and model_path.stat().st_size > 100_000:
        return
    temporary = model_path.with_suffix(model_path.suffix + ".part")
    print(f"[Setup] Downloading {model_path.name} (one time only)...")
    try:
        request = urllib.request.Request(SEGMENTER_URL, headers={"User-Agent": "Nirni-Livewebcam/2.0"})
        with urllib.request.urlopen(request, timeout=timeout) as response, temporary.open("wb") as out:
            while chunk := response.read(1024 * 256):
                out.write(chunk)
        if temporary.stat().st_size < 100_000:
            raise OSError("the downloaded file is too small")
        os.replace(temporary, model_path)
    except (OSError, urllib.error.URLError) as error:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"could not download the body-outline model ({error})") from error


def segmenter_mask(segmenter, frame: np.ndarray, timestamp_ms: int) -> np.ndarray:
    """Person mask (float32, 0..1) from the MediaPipe segmenter, at frame size."""
    import mediapipe as mp

    height, width = frame.shape[:2]
    small = frame
    if width > 384:
        small = cv2.resize(frame, (384, max(1, round(height * 384 / width))),
                           interpolation=cv2.INTER_LINEAR)
    rgb = np.ascontiguousarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
    result = segmenter.segment_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
                                         timestamp_ms)
    if not result.confidence_masks:
        return np.zeros((height, width), dtype=np.float32)
    raw = np.squeeze(result.confidence_masks[0].numpy_view()).astype(np.float32)
    raw = cv2.GaussianBlur(raw, (0, 0), sigmaX=1.2, sigmaY=1.2)
    resized = cv2.resize(raw, (width, height), interpolation=cv2.INTER_LINEAR)
    return np.clip((resized - 0.12) / 0.68, 0.0, 1.0)


def person_mask(segmenter, image, timestamp_ms: int, width: int, height: int) -> np.ndarray:
    """Kept for compatibility: mask from an mp.Image."""
    result = segmenter.segment_for_video(image, timestamp_ms)
    if not result.confidence_masks:
        return np.zeros((height, width), dtype=np.float32)
    raw = np.squeeze(result.confidence_masks[0].numpy_view()).astype(np.float32)
    resized = cv2.resize(raw, (width, height), interpolation=cv2.INTER_LINEAR)
    resized = cv2.GaussianBlur(resized, (0, 0), sigmaX=2.2, sigmaY=2.2)
    return np.clip((resized - 0.12) / 0.68, 0.0, 1.0)


class SegmenterLoader:
    """Loads the body-outline model on a background thread, so a missing
    internet connection never stops the program from starting."""

    def __init__(self, model_path: Path, enabled: bool = True):
        self.model_path = Path(model_path)
        self.segmenter = None
        self.status = "loading" if enabled else "off"
        self.message = ""
        if enabled:
            threading.Thread(target=self._run, name="nirni-segmenter", daemon=True).start()

    def _run(self) -> None:
        try:
            download_segmenter(self.model_path)
        except Exception as error:           # offline: the difference mask takes over
            self.message = str(error)
            self.status = "failed"
            return
        try:
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision

            self.segmenter = vision.ImageSegmenter.create_from_options(
                vision.ImageSegmenterOptions(
                    base_options=python.BaseOptions(model_asset_path=str(self.model_path)),
                    running_mode=vision.RunningMode.VIDEO,
                    output_confidence_masks=True,
                    output_category_mask=False,
                )
            )
            self.status = "ready"
        except Exception as error:           # damaged file or old MediaPipe
            self.message = str(error)
            self.status = "failed"
            self.model_path.unlink(missing_ok=True)   # download a fresh copy next time


class PersonMasker:
    """Gives the person mask for a frame, from the segmenter when it is ready,
    otherwise from the difference against the saved background."""

    def __init__(self, background: BackgroundModel, loader: SegmenterLoader | None = None,
                 threshold: float = 34.0):
        self.background = background
        self.loader = loader
        self.threshold = threshold
        self.errors = 0
        self._ts = 0
        self._bg_key = None
        self._bg_small = None
        self.source = "none"

    @property
    def available(self) -> bool:
        return self.background.ready or (self.loader is not None and self.loader.segmenter is not None)

    def compute(self, frame: np.ndarray, now_ms: int) -> np.ndarray | None:
        segmenter = self.loader.segmenter if self.loader is not None else None
        if segmenter is not None and self.errors < 5:
            try:
                self._ts = max(self._ts + 1, int(now_ms))
                mask = segmenter_mask(segmenter, frame, self._ts)
                self.errors = 0
                self.source = "outline model"
                return mask
            except Exception:
                self.errors += 1
        if not self.background.ready:
            return None
        self.source = "background difference"
        return self._difference_mask(frame)

    def _difference_mask(self, frame: np.ndarray) -> np.ndarray:
        height, width = frame.shape[:2]
        small_w, small_h = max(32, width // 3), max(32, height // 3)
        small = cv2.resize(frame, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
        background = self.background.current()
        key = (id(background), small_w, small_h)
        if key != self._bg_key:
            self._bg_small = cv2.GaussianBlur(
                cv2.resize(background, (small_w, small_h), interpolation=cv2.INTER_AREA), (5, 5), 0)
            self._bg_key = key
        diff = cv2.absdiff(cv2.GaussianBlur(small, (5, 5), 0), self._bg_small)
        b, g, r = cv2.split(diff)
        strength = cv2.max(cv2.max(b, g), r)
        binary = (strength > self.threshold).astype(np.uint8) * 255
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        # Drop specks: keep only blobs that are a real part of the picture.
        count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        if count > 1:
            keep = np.zeros(count, np.uint8)
            keep[1:] = (stats[1:, cv2.CC_STAT_AREA] >= 0.004 * small_w * small_h).astype(np.uint8) * 255
            binary = keep[labels]
        soft = cv2.GaussianBlur(binary.astype(np.float32) / 255.0, (0, 0), 2.0)
        return np.clip(cv2.resize(soft, (width, height), interpolation=cv2.INTER_LINEAR), 0.0, 1.0)


# --------------------------------------------------------------------------
# Vanish styles
# --------------------------------------------------------------------------

def make_dissolve_fields(height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    """Noise map that decides which pixels vanish first (bottom to top)."""
    rng = np.random.default_rng(7391)
    y, x = np.mgrid[0:height, 0:width]
    y_normal = y.astype(np.float32) / max(height - 1, 1)
    x_normal = x.astype(np.float32) / max(width - 1, 1)
    low_noise = rng.random((max(2, height // 10), max(2, width // 10)), dtype=np.float32)
    low_noise = cv2.resize(low_noise, (width, height), interpolation=cv2.INTER_CUBIC)
    fine_noise = rng.random((height, width), dtype=np.float32)
    field_map = (
        (1.0 - y_normal) * 0.72
        + low_noise * 0.17
        + fine_noise * 0.08
        + np.sin(x_normal * 18.0) * 0.025
    )
    return np.clip(field_map, 0.0, 1.0), fine_noise


class Vanisher:
    """Draws you disappearing. progress 0 = fully visible, 1 = fully gone."""

    STYLES = ("dissolve", "pixel", "portal", "glitch")

    def __init__(self, width: int, height: int):
        self.width, self.height = width, height
        self.field, self.sparkle = make_dissolve_fields(height, width)
        self.block = max(8, int(round(width / 64)))
        rows, cols = -(-height // self.block), -(-width // self.block)
        rng = np.random.default_rng(2468)
        sweep = np.linspace(0.0, 1.0, cols, dtype=np.float32)[None, :]
        self.block_field = np.clip(0.72 * rng.random((rows, cols), dtype=np.float32) + 0.28 * sweep, 0, 1)
        # Coordinate grids at half size (the portal is smooth, so half is plenty).
        self._grid = tuple(g[::2, ::2].copy() for g in np.meshgrid(
            np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32)))

    def render(self, style: str, frame: np.ndarray, background: np.ndarray,
               mask: np.ndarray | None, progress: float, now: float = 0.0) -> np.ndarray:
        """Returns a new picture. Neither frame nor background is changed."""
        if style not in self.STYLES:
            style = "dissolve"
        progress = float(np.clip(progress, 0.0, 1.0))
        if mask is None or progress >= 0.999:
            return background.copy()
        margin = 56 if style == "glitch" else 24
        box = mask_bbox(mask, margin)
        if box is None:
            return background.copy()
        out = background.copy()
        getattr(self, "_" + style)(out, frame, background, mask, progress, now, box)
        return out

    # -- dissolve: the original, bottom to top with a glowing edge ----------

    def _dissolve(self, out, frame, background, mask, progress, now, box) -> None:
        x0, y0, x1, y1 = box
        area = np.s_[y0:y1, x0:x1]
        m, field_roi, sparkle = mask[area], self.field[area], self.sparkle[area]
        eased = ease_in_out_cubic(progress)
        survival = smoothstep(eased - 0.055, eased + 0.055, field_roi)
        result = blend(frame[area], background[area], m * survival)
        edge = ((np.abs(field_roi - eased) < 0.018) & (m > 0.48) & (sparkle > 0.54)).astype(np.float32)
        glow = cv2.GaussianBlur(edge, (0, 0), sigmaX=5.0, sigmaY=5.0)
        glow = np.clip(glow * 5.0 + edge, 0.0, 1.0)[:, :, None] * 0.8
        tint = np.array((255, 228, 92), dtype=np.float32)
        out[area] = np.clip(result.astype(np.float32) * (1.0 - glow) + tint * glow, 0, 255).astype(np.uint8)

    # -- pixel: the body breaks into blocks that turn to mosaic, then go ----

    def _pixel(self, out, frame, background, mask, progress, now, box) -> None:
        b = self.block
        height, width = mask.shape[:2]
        x0, y0 = (box[0] // b) * b, (box[1] // b) * b
        x1, y1 = min(width, -(-box[2] // b) * b), min(height, -(-box[3] // b) * b)
        area = np.s_[y0:y1, x0:x1]
        roi_h, roi_w = y1 - y0, x1 - x0
        blocks = self.block_field[y0 // b: -(-y1 // b), x0 // b: -(-x1 // b)]
        eased = ease_in_out_cubic(progress) * 1.25 - 0.1       # lets the last blocks finish
        up = lambda a: cv2.resize(a.astype(np.uint8), (roi_w, roi_h), interpolation=cv2.INTER_NEAREST)

        frame_roi, background_roi, m = frame[area], background[area], mask[area]
        mosaic = cv2.resize(cv2.resize(frame_roi, (blocks.shape[1], blocks.shape[0]),
                                       interpolation=cv2.INTER_AREA),
                            (roi_w, roi_h), interpolation=cv2.INTER_NEAREST)
        mosaic_zone = up((blocks > eased) & (blocks <= eased + 0.22)).astype(bool)
        person = np.where(mosaic_zone[:, :, None], mosaic, frame_roi)
        alpha = m * up(blocks > eased).astype(np.float32)
        result = blend(person, background_roi, alpha)

        edge_blocks = np.abs(blocks - eased) < 0.05
        checker = ((np.indices(blocks.shape).sum(axis=0)) % 2).astype(bool)
        colour = np.where(checker[:, :, None], np.array((255, 235, 80), np.uint8),
                          np.array((255, 80, 230), np.uint8))
        colour = cv2.resize(colour.astype(np.uint8), (roi_w, roi_h), interpolation=cv2.INTER_NEAREST)
        e = (up(edge_blocks).astype(np.float32) * (m > 0.3) * 0.75)[:, :, None]
        out[area] = np.clip(result * (1.0 - e) + colour * e, 0, 255).astype(np.uint8)

    # -- portal: a glowing ring opens at your centre and swallows you -------

    def _portal(self, out, frame, background, mask, progress, now, box) -> None:
        xs, ys = self._grid
        x0, y0, x1, y1 = (v - v % 2 for v in box)
        area = np.s_[y0:y1, x0:x1]
        half = np.s_[y0 // 2:y1 // 2, x0 // 2:x1 // 2]
        roi_h, roi_w = y1 - y0, x1 - x0
        if roi_h < 4 or roi_w < 4:
            return
        m = mask[area]
        moments = cv2.moments(m[::2, ::2])
        if moments["m00"] > 1e-3:
            cx, cy = moments["m10"] / moments["m00"] * 2 + x0, moments["m01"] / moments["m00"] * 2 + y0
        else:
            cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        dx, dy = xs[half] - cx, ys[half] - cy
        radius = max(math.hypot(max(abs(x0 - cx), abs(x1 - cx)), max(abs(y0 - cy), abs(y1 - cy))), 1.0)
        theta = np.arctan2(dy, dx)
        eased = ease_in_out_cubic(progress)
        wobble = 0.035 * np.sin(theta * 6.0 + eased * 9.0) + 0.02 * np.sin(theta * 13.0 - eased * 5.0)
        d = cv2.resize(np.hypot(dx, dy) / radius + wobble, (roi_w, roi_h), interpolation=cv2.INTER_LINEAR)
        mix = cv2.resize(0.5 + 0.5 * np.sin(theta * 3.0 + eased * 12.0), (roi_w, roi_h),
                         interpolation=cv2.INTER_LINEAR)[:, :, None]
        ring_at = eased * 1.12
        survival = smoothstep(ring_at - 0.02, ring_at + 0.02, d)
        result = blend(frame[area], background[area], m * survival)
        ring = np.exp(-(((d - ring_at) / 0.03) ** 2)) * np.clip(m * 1.4, 0.0, 1.0)
        colour = mix * np.array((255, 70, 200), np.float32) + (1.0 - mix) * np.array((255, 230, 80), np.float32)
        g = (ring * 0.95)[:, :, None]
        out[area] = np.clip(result.astype(np.float32) * (1.0 - g) + colour * g, 0, 255).astype(np.uint8)

    # -- glitch: slices tear sideways, colours split, slices drop out --------

    def _glitch(self, out, frame, background, mask, progress, now, box) -> None:
        x0, y0, x1, y1 = box
        area = np.s_[y0:y1, x0:x1]
        roi_h, roi_w = y1 - y0, x1 - x0
        person = frame[area].copy()
        alpha = mask[area].copy()
        rng = np.random.default_rng(int(now * 14.0) % 100000 + 17)
        tear = math.sin(math.pi * progress) ** 1.5
        amplitude = 0.09 * roi_w * tear
        drop = float(np.clip((progress - 0.12) * 1.3, 0.0, 1.0)) ** 1.1
        row = 0
        while row < roi_h:
            band = int(rng.integers(6, 34))
            end = min(roi_h, row + band)
            if rng.random() < drop:
                alpha[row:end] = 0.0
            else:
                shift = int(rng.normal(0.0, amplitude / 2.0)) if rng.random() < 0.65 * tear + 0.05 else 0
                if shift:
                    person[row:end] = np.roll(person[row:end], shift, axis=1)
                    alpha[row:end] = np.roll(alpha[row:end], shift, axis=1)
                split = max(1, abs(shift) // 4) if tear > 0.02 else 0
                if split:
                    person[row:end, :, 2] = np.roll(person[row:end, :, 2], split, axis=1)
                    person[row:end, :, 0] = np.roll(person[row:end, :, 0], -split, axis=1)
            row = end
        if tear > 0.05:
            person[::3] = (person[::3].astype(np.float32) * (1.0 - 0.25 * tear)).astype(np.uint8)
        result = blend(person, background[area], alpha)
        if tear > 0.1:
            for _ in range(int(2 + 6 * tear)):
                y = int(rng.integers(0, roi_h))
                h = int(rng.integers(1, 4))
                bar = (alpha[y:y + h] > 0.5)[:, :, None]
                result[y:y + h] = np.where(bar, np.clip(result[y:y + h].astype(np.int16) +
                                                        np.array((70, 70, 0), np.int16), 0, 255).astype(np.uint8),
                                           result[y:y + h])
        out[area] = result


# --------------------------------------------------------------------------
# Ghost trail and shadow clones
# --------------------------------------------------------------------------

@dataclass(eq=False)
class Cutout:
    crop: np.ndarray            # (h, w, 3) uint8
    alpha: np.ndarray           # (h, w) float32
    origin: tuple[int, int]     # where the crop sits in the picture
    t: float = 0.0
    phase: float = 0.0
    rim: np.ndarray | None = None


def make_cutout(frame: np.ndarray, mask: np.ndarray, now: float, with_rim: bool = False,
                min_area: float = 0.004) -> Cutout | None:
    """Cut the person out of a frame. None if there is nobody in it."""
    box = mask_bbox(mask, 6, threshold=0.25)
    if box is None:
        return None
    x0, y0, x1, y1 = box
    alpha = np.clip((mask[y0:y1, x0:x1] - 0.15) / 0.7, 0.0, 1.0).astype(np.float32)
    if float((alpha > 0.5).sum()) < min_area * mask.shape[0] * mask.shape[1]:
        return None
    rim = None
    if with_rim:
        binary = (alpha > 0.5).astype(np.uint8) * 255
        edge = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, np.ones((5, 5), np.uint8))
        rim = cv2.GaussianBlur(edge.astype(np.float32) / 255.0, (0, 0), 1.5)
    return Cutout(frame[y0:y1, x0:x1].copy(), alpha, (x0, y0), t=now, phase=(now * 1.7) % 6.28, rim=rim)


_TINT_TABLES: dict = {}


def _tint_table(tint: tuple[int, int, int], amount: float) -> np.ndarray:
    """Lookup table that mixes every pixel with a colour. Much faster than
    building a coloured picture for every ghost on every frame."""
    key = (tint, amount)
    table = _TINT_TABLES.get(key)
    if table is None:
        levels = np.arange(256, dtype=np.float32)[:, None]
        table = np.clip(levels * (1.0 - amount) + np.array(tint, np.float32)[None, :] * amount,
                        0, 255).astype(np.uint8).reshape(256, 1, 3)
        _TINT_TABLES[key] = table
    return table


def paste(canvas: np.ndarray, cut: Cutout, dx: int = 0, dy: int = 0, strength: float = 1.0,
          tint: tuple[int, int, int] | None = None, tint_amount: float = 0.0,
          occlude: np.ndarray | None = None, rim_strength: float = 0.0) -> None:
    """Lay a cut-out on the canvas (in place), with optional tint, a glowing
    rim, and the live person in front of it."""
    height, width = canvas.shape[:2]
    ox, oy = cut.origin[0] + int(dx), cut.origin[1] + int(dy)
    ch, cw = cut.alpha.shape
    x0, y0, x1, y1 = max(0, ox), max(0, oy), min(width, ox + cw), min(height, oy + ch)
    if x1 <= x0 or y1 <= y0:
        return
    sx0, sy0 = x0 - ox, y0 - oy
    sx1, sy1 = sx0 + (x1 - x0), sy0 + (y1 - y0)
    crop = cut.crop[sy0:sy1, sx0:sx1]
    alpha = cut.alpha[sy0:sy1, sx0:sx1] * strength
    if occlude is not None:
        alpha = alpha * (1.0 - occlude[y0:y1, x0:x1])
    if tint is not None and tint_amount > 0:
        crop = cv2.LUT(crop, _tint_table(tuple(tint), float(tint_amount)))
    roi = canvas[y0:y1, x0:x1]
    roi[:] = blend(crop, roi, alpha)
    if cut.rim is not None and rim_strength > 0:
        rim = cut.rim[sy0:sy1, sx0:sx1] * rim_strength
        if occlude is not None:
            rim = rim * (1.0 - occlude[y0:y1, x0:x1])
        light = np.empty_like(roi)
        light[:] = (255, 245, 190)
        roi[:] = blend(light, roi, np.clip(rim, 0.0, 1.0))


class GhostEngine:
    """off, a ghost trail (see-through echoes of you a moment ago), or shadow
    clones (frozen copies of you that stay where you left them)."""

    MODES = ("off", "trail", "clones")
    ECHOES = (  # delay s, strength, tint BGR, sideways drift as a share of the width
        (0.18, 0.45, (255, 170, 80), 0.045),
        (0.36, 0.32, (255, 110, 200), -0.045),
        (0.58, 0.20, (160, 255, 130), 0.085),
    )

    def __init__(self, max_clones: int = 3):
        self.mode = "off"
        self.max_clones = max_clones
        self.clones: list[Cutout] = []
        self.history: deque[Cutout] = deque()

    def set_mode(self, mode: str) -> None:
        if mode not in self.MODES:
            raise ValueError(mode)
        if mode == "off":
            self.clones.clear()
        if mode != "trail":
            self.history.clear()
        self.mode = mode

    def cycle(self) -> str:
        self.set_mode(self.MODES[(self.MODES.index(self.mode) + 1) % len(self.MODES)])
        return self.mode

    @property
    def needs_mask(self) -> bool:
        return self.mode == "trail"

    def feed(self, frame: np.ndarray, mask: np.ndarray | None, now: float) -> None:
        if self.mode != "trail" or mask is None:
            return
        if self.history and now - self.history[-1].t < 0.09:
            return
        cut = make_cutout(frame, mask, now)
        if cut is not None:
            self.history.append(cut)
        while self.history and now - self.history[0].t > 1.3:
            self.history.popleft()

    def add_clone(self, frame: np.ndarray, mask: np.ndarray | None, now: float) -> bool:
        if mask is None:
            return False
        cut = make_cutout(frame, mask, now, with_rim=True)
        if cut is None:
            return False
        self.clones.append(cut)
        del self.clones[:-self.max_clones]
        return True

    def clear(self) -> None:
        self.clones.clear()
        self.history.clear()

    def render(self, canvas: np.ndarray, now: float, live_mask: np.ndarray | None = None) -> None:
        width = canvas.shape[1]
        if self.mode == "trail":
            for delay, strength, tint, drift in self.ECHOES:
                target = now - delay
                best = min(self.history, key=lambda c: abs(c.t - target), default=None)
                if best is not None and abs(best.t - target) <= 0.1:
                    paste(canvas, best, dx=int(drift * width), strength=strength, tint=tint,
                          tint_amount=0.4, occlude=live_mask)
        for clone in self.clones:
            bob = int(round(3.0 * math.sin(now * 2.1 + clone.phase)))
            paste(canvas, clone, dy=bob, strength=0.6, tint=(255, 215, 140), tint_amount=0.3,
                  occlude=live_mask, rim_strength=0.9)


# --------------------------------------------------------------------------
# The effect window: blur / pixelate / magnifier inside a rectangle
# --------------------------------------------------------------------------

WINDOW_KINDS = ("blur", "pixelate", "magnify")
WINDOW_COLORS = {"blur": (255, 220, 90), "pixelate": (220, 90, 255), "magnify": (90, 230, 255)}


def apply_window(canvas: np.ndarray, rect, kind: str, zoom: float = 2.0) -> bool:
    """Change the picture inside the rectangle, in place. Returns False when
    the rectangle is too small to do anything."""
    if kind not in WINDOW_KINDS:
        raise ValueError(kind)
    height, width = canvas.shape[:2]
    x0, y0, x1, y1 = rect
    x0, y0, x1, y1 = max(0, int(x0)), max(0, int(y0)), min(width, int(x1)), min(height, int(y1))
    w, h = x1 - x0, y1 - y0
    if w < 8 or h < 8:
        return False
    roi = canvas[y0:y1, x0:x1]
    if kind == "blur":
        small = cv2.resize(roi, (max(1, w // 6), max(1, h // 6)), interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (0, 0), 2.0)
        roi[:] = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    elif kind == "pixelate":
        block = max(6, min(w, h) // 12)
        small = cv2.resize(roi, (max(1, w // block), max(1, h // block)), interpolation=cv2.INTER_AREA)
        roi[:] = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    else:
        cw, ch = max(4.0, w / zoom), max(4.0, h / zoom)
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        sx = int(np.clip(cx - cw / 2.0, 0, max(0.0, width - cw)))
        sy = int(np.clip(cy - ch / 2.0, 0, max(0.0, height - ch)))
        crop = canvas[sy:sy + int(ch), sx:sx + int(cw)]
        roi[:] = cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)
    colour = WINDOW_COLORS[kind]
    cv2.rectangle(canvas, (x0, y0), (x1 - 1, y1 - 1), colour, 3, cv2.LINE_AA)
    cv2.rectangle(canvas, (x0, y0), (x1 - 1, y1 - 1), (255, 255, 255), 1, cv2.LINE_AA)
    tick = max(6, min(22, w // 4, h // 4))
    for (px, py, sx_, sy_) in ((x0, y0, 1, 1), (x1 - 1, y0, -1, 1), (x0, y1 - 1, 1, -1), (x1 - 1, y1 - 1, -1, -1)):
        cv2.line(canvas, (px, py), (px + sx_ * tick, py), colour, 5, cv2.LINE_AA)
        cv2.line(canvas, (px, py), (px, py + sy_ * tick), colour, 5, cv2.LINE_AA)
    return True


# --------------------------------------------------------------------------
# Confetti and smoke
# --------------------------------------------------------------------------

CONFETTI_COLORS = np.array([
    (80, 90, 255), (60, 200, 255), (90, 230, 120), (255, 200, 60),
    (255, 120, 220), (255, 255, 255), (70, 150, 255), (230, 230, 80),
], dtype=np.uint8)


class ParticleSystem:
    """A pool of particles: confetti (kind 0) and smoke puffs (kind 1)."""

    def __init__(self, width: int, height: int, capacity: int = 360, seed: int | None = None):
        self.width, self.height = width, height
        n = capacity
        self.pos = np.zeros((n, 2), np.float32)
        self.vel = np.zeros((n, 2), np.float32)
        self.life = np.zeros(n, np.float32)
        self.kind = np.zeros(n, np.uint8)
        self.color = np.zeros((n, 3), np.uint8)
        self.phase = np.zeros(n, np.float32)
        self.spin = np.zeros(n, np.float32)
        self.size = np.zeros(n, np.float32)
        self.rng = np.random.default_rng(seed)
        self._last: float | None = None

    @property
    def active(self) -> bool:
        return bool((self.life > 0).any())

    @property
    def scale(self) -> float:
        return self.height / 540.0

    def _slots(self, count: int) -> np.ndarray:
        return np.flatnonzero(self.life <= 0)[:count]

    def confetti(self, x: float | None = None, y: float | None = None) -> int:
        """A fountain from (x, y), usually your mouth, plus rain from the top."""
        s, rng = self.scale, self.rng
        x = self.width / 2.0 if x is None else x
        y = self.height * 0.8 if y is None else y
        total = 0
        for count, fountain in ((100, True), (120, False)):
            idx = self._slots(count)
            n = len(idx)
            if not n:
                continue
            if fountain:
                angle = rng.uniform(-math.pi / 2 - 1.0, -math.pi / 2 + 1.0, n)
                speed = rng.uniform(330, 780, n) * s
                self.pos[idx] = (x, y)
                self.vel[idx] = np.stack([np.cos(angle) * speed, np.sin(angle) * speed], axis=1)
            else:
                self.pos[idx] = np.stack([rng.uniform(0, self.width, n),
                                          rng.uniform(-0.3 * self.height, 0, n)], axis=1)
                self.vel[idx] = np.stack([rng.uniform(-40, 40, n), rng.uniform(60, 220, n)], axis=1) * s
            self.life[idx] = rng.uniform(3.0, 4.8, n)
            self.kind[idx] = 0
            self.color[idx] = CONFETTI_COLORS[rng.integers(0, len(CONFETTI_COLORS), n)]
            self.phase[idx] = rng.uniform(0, 6.28, n)
            self.spin[idx] = rng.uniform(5, 12, n)
            self.size[idx] = rng.uniform(0.8, 1.4, n) * s
            total += n
        return total

    def puff(self, x: float, y: float, count: int = 26, spread: float = 90.0) -> int:
        """A burst of smoke, for a shadow clone appearing."""
        s, rng = self.scale, self.rng
        idx = self._slots(count)
        n = len(idx)
        if not n:
            return 0
        angle = rng.uniform(0, 2 * math.pi, n)
        speed = rng.uniform(20, 160, n) * s
        self.pos[idx] = np.stack([x + rng.normal(0, spread / 3, n), y + rng.normal(0, spread / 2, n)], axis=1)
        self.vel[idx] = np.stack([np.cos(angle) * speed, np.sin(angle) * speed - 30 * s], axis=1)
        self.life[idx] = rng.uniform(0.6, 1.1, n)
        self.kind[idx] = 1
        self.size[idx] = rng.uniform(18, 42, n) * s
        self.phase[idx] = 0
        self.color[idx] = (255, 255, 255)
        return n

    def clear(self) -> None:
        self.life[:] = 0

    def update(self, now: float) -> None:
        dt = 0.0 if self._last is None else float(np.clip(now - self._last, 0.0, 0.1))
        self._last = now
        if dt <= 0 or not self.active:
            return
        alive = self.life > 0
        s = self.scale
        confetti = alive & (self.kind == 0)
        puffs = alive & (self.kind == 1)
        self.vel[confetti, 1] = np.minimum(self.vel[confetti, 1] + 520.0 * s * dt, 280.0 * s)
        self.vel[confetti, 0] *= (1.0 - 1.2 * dt)
        self.phase[confetti] += self.spin[confetti] * dt
        self.pos[confetti, 0] += np.sin(self.phase[confetti] * 0.7) * 35.0 * s * dt
        self.vel[puffs] *= (1.0 - 2.5 * dt)
        self.size[puffs] += 70.0 * s * dt
        self.pos[alive] += self.vel[alive] * dt
        self.life[alive] -= dt
        self.life[alive & (self.pos[:, 1] > self.height + 30)] = 0

    def render(self, canvas: np.ndarray) -> None:
        if not self.active:
            return
        height, width = canvas.shape[:2]
        for i in np.flatnonzero((self.life > 0) & (self.kind == 0)):
            x, y = float(self.pos[i, 0]), float(self.pos[i, 1])
            if not (-10 <= x <= width + 10 and -10 <= y <= height + 10):
                continue
            w = (2.0 + 4.0 * abs(math.cos(float(self.phase[i])))) * float(self.size[i])
            h = 5.0 * float(self.size[i])
            c = self.color[i]
            cv2.rectangle(canvas, (int(x - w), int(y - h / 2)), (int(x + w), int(y + h / 2)),
                          (int(c[0]), int(c[1]), int(c[2])), -1)
        smoke = np.flatnonzero((self.life > 0) & (self.kind == 1))
        if smoke.size:
            layer = np.zeros((height // 2 + 1, width // 2 + 1, 3), np.uint8)
            for i in smoke:
                strength = int(255 * min(1.0, float(self.life[i]) * 1.6) * 0.5)
                cv2.circle(layer, (int(self.pos[i, 0] / 2), int(self.pos[i, 1] / 2)),
                           max(1, int(self.size[i] / 2)), (strength,) * 3, -1)
            layer = cv2.GaussianBlur(layer, (0, 0), 5.0)
            layer = cv2.resize(layer, (width, height), interpolation=cv2.INTER_LINEAR)
            canvas[:] = cv2.add(canvas, layer)
