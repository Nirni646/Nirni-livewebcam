"""Nirni Livewebcam - gesture magic for your webcam.

Copyright (c) 2026 Sidagam Nirni Naidu. All rights reserved.
Use of this file is governed by the LICENSE file in this folder.

What it does:
  * Touch your nose and you vanish (four styles). Touch it again to come back.
  * Touch your forehead for draw tools: join the index fingertips of both
    hands to draw, and the stroke becomes a clean shape. Pick colours, the
    eraser, undo and redo by pointing at the buttons on the screen edge.
  * Close a fist over a shape to carry it; two fists resize and rotate it.
  * Touch your chin for the effect window: point with both index fingers and
    the box between them is blurred, pixelated or magnified.
  * Ghost trail and shadow clones, wink to undo, a big smile for confetti,
    photos and video recording.

Everything runs on your own computer. Nothing is uploaded. Photos and videos
are saved only when you ask for them, in the captures folder.
"""

from __future__ import annotations

import argparse
import platform
import sys
import time
from contextlib import ExitStack
from pathlib import Path

import cv2
import numpy as np

from air_canvas import is_pointing

__version__ = "2.0.0"
APP_NAME = "Nirni Livewebcam"
AUTHOR = "Sidagam Nirni Naidu"
WINDOW_NAME = "Nirni Livewebcam"

APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "models"


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------

def positive_fps(value: str) -> float:
    fps = float(value)
    if not np.isfinite(fps) or not 1 <= fps <= 120:
        raise argparse.ArgumentTypeError("FPS must be between 1 and 120.")
    return fps


def picture_size(value: str) -> int:
    size = int(value)
    if not 160 <= size <= 4096:
        raise argparse.ArgumentTypeError("size must be between 160 and 4096.")
    return size


def fraction(value: str) -> float:
    number = float(value)
    if not np.isfinite(number) or not 0.03 <= number <= 1.5:
        raise argparse.ArgumentTypeError("distance must be between 0.03 and 1.5.")
    return number


def seconds(value: str) -> float:
    number = float(value)
    if not np.isfinite(number) or not 0.05 <= number <= 5.0:
        raise argparse.ArgumentTypeError("hold time must be between 0.05 and 5 seconds.")
    return number


def parse_args(argv=None) -> argparse.Namespace:
    from effects import Vanisher

    parser = argparse.ArgumentParser(
        description=f"{APP_NAME} {__version__}: gesture magic for your webcam."
    )
    parser.add_argument("--camera", type=int, default=0, help="webcam index (default: 0)")
    parser.add_argument("--width", type=picture_size, default=1280, help="video width (default: 1280)")
    parser.add_argument("--height", type=picture_size, default=720, help="video height (default: 720)")
    parser.add_argument("--fast", action="store_true",
                        help="for slower computers: 640x360 picture and lighter tracking")
    parser.add_argument("--fullscreen", action="store_true", help="start in fullscreen")
    parser.add_argument("--no-preview", action="store_true",
                        help="no preview window, only the virtual camera")
    parser.add_argument("--no-mirror", action="store_true", help="do not mirror the webcam")
    parser.add_argument("--raw-video", action="store_true",
                        help="do not ask the camera for MJPG (try this if the picture is broken)")
    parser.add_argument("--fps", type=positive_fps, default=30.0,
                        help="FPS of the virtual camera and of recordings (default: 30)")
    parser.add_argument(
        "--virtual-camera", action=argparse.BooleanOptionalAction, default=None,
        help="require a virtual camera; by default it is used when available. "
             "--no-virtual-camera turns it off",
    )
    parser.add_argument("--virtual-backend", choices=("obs", "unitycapture", "v4l2loopback"),
                        help="virtual camera driver to use")
    parser.add_argument("--virtual-device", help="name of the virtual camera device")
    parser.add_argument("--style", choices=Vanisher.STYLES, default="dissolve",
                        help="how you vanish (T changes it while running)")
    parser.add_argument("--hold", type=seconds, default=0.45,
                        help="seconds the finger must stay on the nose, forehead or chin (default: 0.45)")
    parser.add_argument("--touch-distance", type=fraction, default=0.22,
                        help="how close the fingertip must be to the nose, as a fraction "
                             "of the face width (default: 0.22, bigger is more forgiving)")
    parser.add_argument("--forehead-distance", type=fraction, default=0.20,
                        help="same as --touch-distance, but for the forehead (default: 0.20)")
    parser.add_argument("--chin-distance", type=fraction, default=0.20,
                        help="same as --touch-distance, but for the chin (default: 0.20)")
    parser.add_argument("--pen-join", type=fraction, default=0.55,
                        help="how close the two index fingertips must be to start drawing, "
                             "in palm sizes (default: 0.55, bigger is more forgiving)")
    parser.add_argument("--no-face-gestures", action="store_true",
                        help="turn off wink-to-undo and smile-for-confetti")
    parser.add_argument("--smile-level", type=float, default=0.55,
                        help="how wide the smile must be for confetti, 0 to 1 (default: 0.55)")
    parser.add_argument("--output-dir", type=Path, default=APP_DIR / "captures",
                        help="where photos and videos are saved (default: captures/)")
    parser.add_argument("--no-outline-model", action="store_true",
                        help="never download the body-outline model, use background difference only")
    parser.add_argument("--debug", action="store_true", help="print detection info in the terminal")
    parser.add_argument("--self-test", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--video-file", help=argparse.SUPPRESS)
    parser.add_argument("--frames", type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    given = argv if argv is not None else sys.argv[1:]
    if args.fast:
        if not any(a == "--width" or a.startswith("--width=") for a in given):
            args.width = 640
        if not any(a == "--height" or a.startswith("--height=") for a in given):
            args.height = 360
    args.detect_width = 320 if args.fast else 512
    args.smile_level = min(max(args.smile_level, 0.1), 0.95)
    return args


# --------------------------------------------------------------------------
# Camera
# --------------------------------------------------------------------------

def open_camera(index, width: int, height: int, fps: float = 30.0,
                raw_video: bool = False) -> cv2.VideoCapture:
    if isinstance(index, str):                       # a video file (testing)
        capture = cv2.VideoCapture(index)
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open the video file {index}.")
        return capture
    if platform.system() == "Windows":
        capture = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not capture.isOpened():
            capture.release()
            capture = cv2.VideoCapture(index)
    else:
        capture = cv2.VideoCapture(index)

    if not capture.isOpened():
        raise RuntimeError(
            f"Webcam {index} is not available. Close other apps that use the "
            "camera, or try --camera 1."
        )
    if not raw_video:
        # MJPG is what lets most webcams deliver 720p at a full 30 FPS over USB.
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    capture.set(cv2.CAP_PROP_FPS, fps)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return capture


def open_virtual_camera(args: argparse.Namespace, width: int, height: int):
    if args.virtual_camera is False:
        return None
    try:
        import pyvirtualcam

        camera = pyvirtualcam.Camera(
            width=width, height=height, fps=args.fps,
            fmt=pyvirtualcam.PixelFormat.BGR,
            backend=args.virtual_backend, device=args.virtual_device,
        )
    except (ImportError, RuntimeError, OSError, ValueError) as error:
        message = (
            "Virtual camera not available. Install OBS Studio (it includes the "
            "OBS Virtual Camera driver). If OBS is already open, stop its own "
            f"virtual camera first. Details: {error}"
        )
        if args.virtual_camera is True or args.virtual_backend or args.virtual_device:
            raise RuntimeError(message) from error
        print(f"[Camera] {message}\n[Camera] Continuing with the preview window only.")
        return None
    print(f"[Camera] Virtual camera is on: {camera.device} "
          f"({width}x{height}, up to {args.fps:g} FPS)")
    return camera


def send_virtual_frame(camera, frame: np.ndarray) -> None:
    if frame.shape[:2] != (camera.height, camera.width):
        frame = cv2.resize(frame, (camera.width, camera.height), interpolation=cv2.INTER_LINEAR)
    camera.send(np.ascontiguousarray(frame))


# --------------------------------------------------------------------------
# Main loop
# --------------------------------------------------------------------------

def set_fullscreen(enabled: bool) -> None:
    mode = cv2.WINDOW_FULLSCREEN if enabled else cv2.WINDOW_NORMAL
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, mode)


def key_to_char(key: int) -> str | None:
    """cv2.waitKey code -> lower-case character, so Caps Lock changes nothing."""
    if key == -1:
        return None
    code = key & 0xFF
    if code in (27, 13, 10):
        return chr(code)
    if 32 <= code < 127:
        return chr(code).lower()
    return None


def window_closed() -> bool:
    try:
        return cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:
        return True


def self_test() -> int:
    """Check that the bundled models load, without opening a webcam."""
    from tracking import create_detector

    detector = create_detector()
    detector.close()
    print(f"[{APP_NAME}] Self-test passed: face and hand models loaded.")
    return 0


def run_camera(args: argparse.Namespace, resources: ExitStack) -> int:
    from effects import SEGMENTER_NAME, SegmenterLoader
    from media import CameraStream
    from studio import Studio
    from tracking import DetectionWorker, create_detector
    from ui import draw_hud

    source = args.video_file if args.video_file else args.camera
    capture = open_camera(source, args.width, args.height, args.fps, args.raw_video)
    stream = CameraStream(capture, mirror=not args.no_mirror)
    resources.callback(stream.close)
    detector = create_detector(not args.no_face_gestures, args.detect_width, MODEL_DIR)
    worker = DetectionWorker(detector)
    resources.callback(worker.close)
    loader = SegmenterLoader(MODEL_DIR / SEGMENTER_NAME, enabled=not args.no_outline_model)

    first = stream.read(0, timeout=8.0)
    if first is None:
        raise RuntimeError("The webcam did not deliver a picture. Try --camera 1 or --raw-video.")
    height, width = first[0].shape[:2]
    print(f"[{APP_NAME}] Camera picture: {width}x{height}")

    studio = Studio(width, height, hold=args.hold, nose_distance=args.touch_distance,
                    forehead_distance=args.forehead_distance, chin_distance=args.chin_distance,
                    pen_join=args.pen_join, face_gestures=not args.no_face_gestures,
                    smile_level=args.smile_level, style=args.style,
                    output_dir=args.output_dir, loader=loader, fps=args.fps)
    resources.callback(studio.close)
    studio.phase_started = time.perf_counter()
    virtual_camera = open_virtual_camera(args, width, height)
    if virtual_camera is not None:
        resources.callback(virtual_camera.close)

    preview = not args.no_preview
    fullscreen = args.fullscreen
    if preview:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, min(width, 1280), min(height, 720))
        resources.callback(cv2.destroyAllWindows)
        if fullscreen:
            set_fullscreen(True)

    print(f"[{APP_NAME}] Step out of the frame. Capturing the background (press H for help).")
    last_id = 0
    fps = 0.0
    shown = 0
    last_tick = time.perf_counter()
    last_report = last_tick
    while True:
        item = stream.read(last_id, timeout=1.0)
        if item is None:
            if preview and shown > 5 and window_closed():
                break
            continue
        frame, last_id = item
        worker.submit(frame, last_id)
        now = time.perf_counter()
        display = studio.process(frame, worker.latest(), now)

        if virtual_camera is not None:
            try:
                send_virtual_frame(virtual_camera, display)
            except Exception as error:     # OBS closed, driver reset: keep the program alive
                print(f"[Camera] Virtual camera stopped ({error}). Continuing with the preview only.")
                try:
                    virtual_camera.close()
                except Exception:
                    pass
                virtual_camera = None

        tick = time.perf_counter()
        fps = 0.92 * fps + 0.08 / max(tick - last_tick, 1e-4) if shown else 1.0 / max(tick - last_tick, 1e-4)
        last_tick = tick
        shown += 1

        if args.debug and tick - last_report >= 1.0:
            last_report = tick
            obs = studio._latest_obs
            ratios = [d.closest_ratio for d in (studio.gestures.nose, studio.gestures.forehead,
                                                studio.gestures.chin)]
            fmt = lambda r: "n/a" if r is None else f"{r:.2f}"
            print(f"[Debug] {fps:.0f} fps, detect {worker.detect_ms:.0f} ms, camera {stream.measured_fps:.0f} fps | "
                  f"faces={int(obs.face_found)} hands={len(obs.hands)} pointing={sum(is_pointing(h) for h in obs.hands)} | finger-to: nose={fmt(ratios[0])} "
                  f"(needs <= {args.touch_distance:.2f}) forehead={fmt(ratios[1])} "
                  f"(<= {args.forehead_distance:.2f}) chin={fmt(ratios[2])} (<= {args.chin_distance:.2f}) | "
                  f"smile={obs.smile if obs.smile is None else round(obs.smile, 2)} "
                  f"mask={studio.masker.source} phase={studio.phase} shapes={len(studio.board.shapes)}")

        if args.frames and shown >= args.frames:
            print(f"[{APP_NAME}] {shown} frames done, average {fps:.1f} FPS.")
            break
        if preview:
            cv2.imshow(WINDOW_NAME, draw_hud(studio, display, now, fps))
            ch = key_to_char(cv2.waitKey(1))
            if shown > 5 and window_closed():
                break
            if ch is not None:
                action = studio.on_key(ch, time.perf_counter())
                if action == "quit":
                    break
                if action == "fullscreen":
                    fullscreen = not fullscreen
                    set_fullscreen(fullscreen)
        if virtual_camera is not None:
            virtual_camera.sleep_until_next_frame()
    return 0


def run(argv=None) -> int:
    args = parse_args(argv)
    if args.self_test:
        return self_test()
    with ExitStack() as resources:
        return run_camera(args, resources)


def main() -> int:
    try:
        return run()
    except KeyboardInterrupt:
        return 0
    except Exception as error:
        print(f"\n[{APP_NAME}] ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
