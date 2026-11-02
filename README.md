# Nirni Livewebcam 2.0

Gesture magic for your webcam.

- **Touch your nose** and you vanish (4 styles). Touch it again and you come back.
- **Touch your forehead** for draw tools: draw in the air, pick colours, erase, undo, redo.
- **Touch your chin** for the effect window: a box between your two index fingers that
  blurs, pixelates or magnifies whatever is inside it.
- **Ghost trail and shadow clones** of yourself.
- **Wink to undo. Smile big for confetti.**
- **Two fists resize and rotate** a shape.
- **Photos and video recording** of exactly what your call sees.

It watches your face and your hands. It runs completely on your own computer.
Nothing is uploaded or sent anywhere. A photo or a video is saved only when you ask
for it, into the `captures` folder next to the program.

Made by **Sidagam Nirni Naidu**. All rights reserved, see [LICENSE](LICENSE).

## What you need

- Windows with Python 3.10 or newer (64-bit). When installing Python, tick
  **Add python.exe to PATH**.
- A webcam.
- Internet for the first run only (packages and one small model). Without internet the
  program still works: it just uses a slightly simpler way to find your outline.
- Optional: [OBS Studio](https://obsproject.com/) to use the effects in Zoom, Meet,
  Discord and similar apps.

## Run it

Double-click `run.bat`. The first time it creates a local Python environment and
installs everything. Slower computer? Run `run.bat --fast`.

Or by hand, from inside this folder:

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python nirni_livewebcam.py
```

## First steps

1. When the window opens, **step out of the frame**. The program waits until the picture
   is really empty and still (it checks for your face, your hands and movement), then
   saves the background. Press `ENTER` to save immediately anyway.
2. Step back in. Press `H` any time to show the help on the screen.

## Controls

### Touch your face (hold about half a second)

| Touch            | What happens                                              |
|------------------|-----------------------------------------------------------|
| nose tip         | vanish / come back (also reverses a vanish half way)      |
| middle of forehead | draw tools on / off                                     |
| chin             | effect window on / off                                    |

### Face expressions

| You do                       | What happens         |
|------------------------------|----------------------|
| wink (one eye, hold a moment)| undo                 |
| wide smile, mouth closed-ish, held | confetti from your mouth |

A normal blink, talking and small smiles do not trigger these, and confetti has a
6 second pause. Turn both off with `S` or `--no-face-gestures`.

### Drawing (draw tools on)

- **Join the index fingertips of both hands** and move: that is the pen. Pull your hands
  apart to finish. The stroke becomes a circle, ellipse, triangle, square, rectangle,
  straight line, or a smooth curve.
- **Pick colours, eraser, undo, redo by pointing.** In draw mode buttons appear on the
  screen edges (in your preview only, never in the call). Point at one with your index
  finger, fingers curled, and wait a moment: a bar fills and the button is pressed.
- **Eraser:** choose `ERASE`, then touch shapes with a pointing finger.
- **Rainbow** is a colour too. `0` goes back to "a new colour for every shape".

### Moving shapes

- **Open hand over a shape, then close it into a fist** to catch it. Move, then open the
  hand to leave it.
- **Resize and rotate:** while one fist holds a shape, close your **second fist anywhere**.
  Move the fists apart or together to resize, turn them to rotate. Open the second fist to
  go back to carrying with one hand. Every move, resize and rotate is one undo step.

### Effect window (chin touch, `W`, or the BLUR / PIX / ZOOM buttons)

Point with the index fingers of **both hands** at two opposite corners. The box between
them gets the effect. Both fingers must be pointing (other fingers curled) and apart.

| Effect | Use                                    |
|--------|----------------------------------------|
| BLUR   | hide something behind you or on a desk |
| PIX    | pixelate (mosaic)                      |
| ZOOM   | a 2x magnifier box                     |

The box is part of the picture your call sees. With one hand only, the buttons work.

### Ghost mode (`G`, GHOST button)

- **Ghost trail:** see-through echoes of you a moment ago follow you around.
- **Shadow clones:** `N` (or CLONE) leaves a frozen see-through copy of you with a puff of
  smoke. Walk away, it stays. Up to 3. You stand in front of your clones, not behind them.
- Pressing `G` again cycles trail, clones, off.

### Vanish styles (`T` changes, or `--style`)

`dissolve` (the original, feet first), `pixel` (breaks into blocks), `portal` (a glowing
ring opens in the middle of you), `glitch` (slices tear, colours split).

### Photos and video

- `P` takes a photo now. The SNAP button takes one after a 3 second countdown, so your
  pointing hand is out of the picture.
- `M` or REC starts and stops recording. The video has the same clean picture as the
  virtual camera, at a steady frame rate. **There is no sound** (the program never
  touches your microphone).
- Files go to `captures/` (`--output-dir` changes it).

### Keys

| Key | Does | Key | Does |
|---|---|---|---|
| `SPACE` | new background | `Z` / `Y` | undo / redo |
| `ENTER` | save background now | `C` | clear all (undoable) |
| `V` / `R` | vanish / return | `1`-`7` | colours |
| `T` | vanish style | `8` / `0` | rainbow / auto colour |
| `D` | draw tools | `[` `]` | pen size |
| `E` | eraser | `P` | photo |
| `W` | effect window (blur, pix, zoom, off) | `M` | record on / off |
| `G` | ghost (trail, clones, off) | `B` | confetti |
| `N` | new shadow clone | `S` | wink / smile on / off |
| `H` | help | `F` | fullscreen |
| `Q` / `ESC` | quit (or close the window) | | |

Caps Lock does not matter.

## Options

```
--fast                  640x360 and lighter tracking, for slower computers
--width 1920 --height 1080   bigger picture (default 1280x720)
--fps 60                virtual camera and recording FPS (default 30)
--camera 1              another webcam
--style glitch          start with another vanish style
--no-preview            no window, only the virtual camera
--no-face-gestures      turn off wink and smile
--smile-level 0.7       harder to trigger confetti (0 to 1)
--hold 0.8              hold touches longer
--touch-distance 0.30   more forgiving nose touch (also --forehead-distance, --chin-distance)
--pen-join 0.8          fingertips may be further apart to draw
--output-dir D:\clips   where photos and videos go
--raw-video             do not ask the camera for MJPG (if the picture looks broken)
--no-outline-model      never download the outline model
--no-mirror             do not mirror the picture
--debug                 print FPS, timings and detection numbers every second
```

If a touch triggers too easily raise `--hold` or lower the distance. If it never
triggers, run with `--debug` and watch the `finger-to` numbers while you touch: they must
drop below the `needs <=` value.

## Speed

Version 2 is much faster than version 1:

- The camera, the face and hand tracking, and the drawing run on separate threads, so
  slow tracking never makes the video stutter and the picture is always the newest one.
- The camera is asked for MJPG, which is what lets most webcams deliver 720p at a full
  30 FPS.
- Tracking sees a shrunken copy of the picture. Effects only work on the small area they
  change, not the whole frame. The FPS is shown at the bottom right of the preview.

## Using it in video calls

If OBS Studio is installed, the program sends its picture to **OBS Virtual Camera**
automatically. In your call app, pick OBS Virtual Camera. Do not start the virtual
camera from inside OBS at the same time. `run_virtual_camera.bat` makes the virtual camera
mandatory. If the virtual camera stops while you are using the program, it keeps running
with the preview only.

## What was fixed in 2.0

- Starting without internet no longer fails: the outline model loads in the background
  and a difference-from-background mask is used until it is ready (or forever, offline).
- Background capture waits until the picture is empty and still, instead of a blind
  countdown that could save a ghost of you.
- The saved background follows the camera's auto-exposure and white balance, so no
  silhouette appears when you vanish after the lighting changes.
- One hand detected twice can no longer draw by "joining its own fingertips".
- Clicking the window's X now really closes the program.
- Caps Lock no longer breaks the keys.
- A virtual camera error, or a short webcam glitch, no longer crashes the program.
- A nose touch can now reverse a vanish half way. Shapes stay visible while recalibrating.
- If the camera changes picture size, shapes, background and recordings adapt.
- Undo now undoes any action (add, move, resize, erase, clear), with redo.
- Option values are checked, so `--hold 0` and similar give a clear message.

## Troubleshooting

- **`python` is not recognised:** reinstall Python, tick *Add python.exe to PATH*, open a
  new cmd window.
- **`pip install` fails on mediapipe:** use 64-bit Python 3.10, 3.11 or 3.12.
- **Black screen or "webcam not available":** close other apps using the camera, try
  `--camera 1`, or `--raw-video`.
- **A ghost of you stays after vanishing:** press `SPACE` and step fully out.
- **Wink or smile triggers by accident:** press `S`, or raise `--smile-level`.
- **A button does not press:** curl the other fingers, keep the index fingertip on the
  button until the bar fills.
- **The window effect does not appear:** both hands need to point (index out, others
  curled) and be apart. Use `--debug` to see how many hands are seen.
- **Choppy video:** use `--fast`, close other heavy apps, and make sure the room is bright
  (dark rooms make webcams drop to low frame rates).
- **The pen will not go down:** raise `--pen-join`, keep both hands fully visible.

## How it works

- **Tracking:** MediaPipe Face Landmarker (with expression scores for wink and smile) and
  Hand Landmarker, on a background thread.
- **Your outline:** MediaPipe selfie segmentation when available, otherwise the difference
  between the live picture and the saved background.
- **Touches:** a fingertip must stay near the nose, forehead or chin (relative to your face
  width) for a short time. Gestures are ignored while you draw or carry a shape.
- **Shapes:** each stroke is classified (closed? round? corners? straight?) and stored as a
  vector shape with position, size and angle, so undo, resize and rotate are exact.

## Project files

```
nirni_livewebcam.py            the program: options, camera loop
studio.py                      modes, gestures and effects put together
air_canvas.py                  pen, shape recognition, shape board, undo / redo
gestures.py                    touches, wink, smile, tool dock, effect window
tracking.py                    face and hand tracking on a worker thread
effects.py                     vanish styles, ghosts, window effects, confetti
media.py                       camera thread, recorder, photos
ui.py                          the hints you see in the preview
run.bat, run_virtual_camera.bat   launchers
requirements.txt               Python packages
models/                        face and hand models (the outline model downloads here)
tests/                         unit tests
LICENSE, NOTICE, CERTIFICATE_OF_AUTHORSHIP.md
```

Run the tests with `python -m unittest discover -s tests`.

## Author

Sidagam Nirni Naidu

Copyright (c) 2026 Sidagam Nirni Naidu. All rights reserved.
