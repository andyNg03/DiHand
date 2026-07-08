# DiHand

Draw diagrams with your hands. DiHand tracks your hands through a webcam and turns gestures into shapes on a canvas — no mouse, no keyboard. Form an "L" with both hands to sketch a square, frame a circle with your fingers, point two index fingers to draw an arrow between them, then grab and rearrange shapes in mid-air. When you're happy with the result, hold a fist to export it as a PNG or an editable `.excalidraw` file.

It's a small, self-contained experiment in gesture interaction: hand-tracking demos and diagramming tools both exist, but stitching them together into a gesture-driven canvas turned out to be an interesting design problem in its own right.

Built with [MediaPipe](https://developers.google.com/mediapipe) for hand landmark detection and [Pygame](https://www.pygame.org/) for the canvas and event loop.

## Demo

Run the app, hold up a hand, and you'll see a live skeleton overlay on each detected hand, a per-hand gesture label, and the current mode in the top-left corner. Everything is driven by the poses in the [gesture reference](#gesture-reference) below.
<p align="center">
  <img src="assets/DiHand_Demo.gif" alt="DiHand demo" width="600">
</p>

## Requirements

- **Python 3.13** (tested). MediaPipe and Pygame both ship prebuilt wheels for it, so nothing has to be compiled. The code uses no version-specific features, so other recent 3.x versions likely work too — 3.13 is just the one that's verified.
- A webcam
- The MediaPipe hand model (`hand_landmarker.task`) ships in this repo, so no separate download is needed.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install mediapipe pygame
```

That's it — on a supported Python this installs entirely from prebuilt wheels, with no compiler or system libraries required. MediaPipe pulls in OpenCV and NumPy as dependencies; DiHand uses OpenCV for webcam capture and NumPy for coordinate math.

> On import you may see `objc[...]: Class SDL... is implemented in both ...` warnings on macOS. OpenCV and Pygame each bundle their own copy of SDL2, and the runtime notices the duplicate. They're harmless — the app runs correctly.

<details>
<summary>Running on Python 3.14</summary>

3.14 is new enough that Pygame may not have a prebuilt wheel yet, so `pip` will try to build it from source — which needs SDL2 installed **first**, or the install fails with a build error. On macOS:

```bash
brew install sdl2 sdl2_image sdl2_mixer sdl2_ttf   # install before `pip install pygame`
```

Everything else is the same. If you're not tied to 3.14, using a 3.9–3.13 interpreter avoids this entirely.
</details>

## Usage

```bash
source venv/bin/activate
python app.py
```

Press `q` or close the window to quit.

To print live performance stats to the terminal — frames per second, per-frame time, and MediaPipe detection time — add `--profile`:

```bash
python app.py --profile
```

It reports roughly once a second and is off by default, so a normal run carries no overhead.

DiHand is **mode-based**: a gesture only means what the current mode says it means, which keeps poses from being misread. You explicitly toggle modes, rather than the app guessing your intent every frame.

- **Idle** — the default. Nothing is drawn. You can select and move existing shapes here, or hold a fist to export.
- **Draw** — you're creating a shape. A live preview follows your hands; committing it drops you back to Idle.

Selection and dragging are **reactive**, not a separate mode: any time you're in Idle, a peace sign over a shape highlights it, and closing those fingers grabs it.

### Typical flow

1. **Prayer hands** (both palms up, held close with a small gap) → enter **Draw** mode.
2. Form a shape gesture — L-frames for a square, a converging three-finger frame for a diamond, a two-hand claw for a circle, or two pointing fingers for an arrow. A translucent preview appears.
3. **Two fists** → commit the shape to the canvas and return to Idle. A **single fist** cancels and exits without committing.
4. Back in Idle, hold a **peace sign** over a shape to highlight it, **close the two fingers** to grab it, and move your hand to drag it. Re-open to drop.
5. To **delete**, drag a grabbed shape into the border zone at any screen edge (a pink border warns you) and release.
6. **Hold a single fist for 3 seconds** in Idle to export. A filling green ring shows progress; on completion you get a timestamped `.png` and `.excalidraw` pair in the working directory.

## Gesture reference

| Gesture | Hands | Meaning |
|---|---|---|
| Prayer hands (palms up, small gap, fingertips aligned) | 2 | Enter Draw mode |
| L-frames (open palm + thumb out, both hands) | 2 | Preview a **square** |
| Three-finger frame (index+middle+ring extended, tips converging) | 2 | Preview a **diamond** |
| Two-hand claw (all four fingers half-curled, tips clustered) | 2 | Preview a **circle** |
| Two index fingers pointing | 2 | Preview an **arrow** (physical left hand = source, right = destination) |
| Two fists | 2 | Commit the previewed shape → Idle |
| Single fist | 1 | Cancel/exit Draw; or hold 3s in Idle to **export** |
| Peace sign, fingers apart | 1 | Highlight the shape under your fingers |
| Peace sign, fingers closed | 1 | Grab & drag the highlighted shape |

The diamond gesture is internally still called `triangle_frame` — it describes the hand pose (a triangular "frame"), even though the shape it produces is a diamond. Diamonds are used instead of triangles so shapes round-trip losslessly through Excalidraw, which has native diamond and rectangle elements but no triangle.

## How it works

The whole pipeline is a real-time loop: read a webcam frame, mirror it, ask MediaPipe for up to two hands' worth of 21 landmarks, classify the pose from those landmarks, update app state, and redraw the window.

Gesture recognition is entirely **heuristic** — no custom ML model. Each pose is a set of rules over landmark geometry: a finger counts as "extended" or "curled" based on the ratio of its fingertip-to-wrist distance against its knuckle-to-wrist distance; two-hand gestures add checks on the relative position of the two hands. Distances are computed in 3D (including MediaPipe's depth estimate) because a palm tilted away from the camera foreshortens in 2D and throws the ratios off.

### Modules

| File | Responsibility |
|---|---|
| `app.py` | The driver: sets up MediaPipe and the Pygame window, runs the frame loop, wires everything together. |
| `gestures.py` | Landmark math and the gesture classifier — single-hand (`classify`) and two-hand (`classify_two_hand`). |
| `shape.py` | The shape model: building shapes from landmarks, drawing them, hit-testing, and translating them. |
| `selection.py` | Reactive selection state — hover, grab, drag, and delete-by-drag. |
| `export.py` | Rendering the canvas to PNG and serializing it to `.excalidraw` JSON. |

Shapes are plain dictionaries (`{"type": "square", "center": (x, y), "half": 55}` and so on), so any part of the app can build, move, draw, or serialize them without a class hierarchy.

## Known limitations

These are inherent to the approach or deliberately out of scope; a few are on the list to revisit.

- **Prayer hands (Draw entry) is fiddly.** MediaPipe can't cleanly separate two hands that touch or overlap — it merges them into one detection. So the trigger pose needs a visible gap between the palms, which makes it less "prayer-like" than ideal. A less occlusion-prone trigger is the main candidate for a redesign.
- **The circle gesture is awkward.** The "claw" pose sits in a narrow band of finger-curl — deliberate enough to avoid false-firing from a closing fist, but not a natural "I'm drawing a circle" motion.
- **Arrows can't be rotated.** Moving an arrow translates both endpoints together; there's no way to re-anchor a single end or rotate it. Angled arrows have to be deleted and redrawn.
- **Arrow direction relies on MediaPipe's handedness labels,** which can flip during crossed-hand or heavily rotated poses, momentarily reversing the arrow.
- **Exports land in the working directory** with timestamped names — no save dialog or configurable path.

Out of scope for this version: text labels on shapes, multi-select, and undo/redo.

## Acknowledgements

This project was built with AI assistance (Claude). The design decisions, gesture vocabulary, and direction are mine; I used AI as a pair-programming and rubber-ducking partner along the way.
