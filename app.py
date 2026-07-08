"""
DiHand — gesture-driven diagramming.

Structure: setup (module-level code below) → render loop → finally-block cleanup.
The loop reads webcam frames, asks MediaPipe for hand landmarks, classifies
gestures, updates app state, then redraws the whole window each frame.
Press 'q' or close the window to quit.
"""

# Stdlib.
import math                              # sqrt + arc angle math for the progress circle
import sys                               # read argv for the --profile flag
import time                              # frame timestamps + export hold timer

# Third-party.
import cv2                               # webcam capture, color/flip ops
import mediapipe as mp                   # hand landmark detection
import pygame                            # window, drawing, events, fonts

# MediaPipe's Tasks API is split into two packages — we import them once
# here and use the short aliases everywhere else.
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# local imports**
# Our own gesture module. Landmark indices are int constants (e.g. WRIST=0)
# so callers can write hand[WRIST] instead of hand[0]. Only WRIST is used
# directly here (Stage 9c per-hand labels); shape.py and selection.py
# import the landmarks they need separately.
from gestures import (
    WRIST, classify, classify_two_hand
)
# Shape module. All shape model + drawing + hit-testing + translation lives
# in shape.py; we import only what the driver uses directly. Other modules
# (selection.py, export.py) import from shape independently.
from shape import (
    # constants the driver still references
    PREVIEW_ALPHA,
    # public builders + dispatch
    GESTURE_BUILDERS,
    # public drawing
    draw_shape, init_preview_surface,
)
# Selection / grab state — reactive hover + drag/delete state machine.
# Class definition + comments live in selection.py; we just import and
# instantiate here. One instance for the lifetime of the app.
from selection import Selection
selection = Selection()
# Export — PNG (rasterized) + .excalidraw (editable JSON).
# The whole machinery lives in export.py; the driver just calls
# export_diagram(shapes, w, h) when Stage 8's hold timer expires.
from export import export_diagram


# Initializes pygame's subsystems (display, font, etc.). Must run before
# we touch any pygame functionality below.
pygame.init()


# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------

# Path to the MediaPipe model file. Loaded once at landmarker creation time.
MODEL_PATH = "hand_landmarker.task"

# Skeleton overlay: pairs of (start, end) landmark indices that get connected
# by a line. Anatomically grouped — thumb chain, then each finger chain, then
# the palm base. 0 = wrist; 4/8/12/16/20 = fingertips.
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),         # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),         # index
    (5, 9), (9, 10), (10, 11), (11, 12),    # middle
    (9, 13), (13, 14), (14, 15), (15, 16),  # ring
    (13, 17), (17, 18), (18, 19), (19, 20), # pinky
    (0, 17),                                # palm base (wrist → pinky MCP)
]


# ----------------------------------------------------------------------------
# Mode state machine
# ----------------------------------------------------------------------------
#
# Modes are durable, so we don't flip on a single frame. The trigger gesture
# must hold for DEBOUNCE_FRAMES consecutive frames; one miss resets the count.
# All of this — labels, rules, debounce, and runtime state — lives in
# ModeMachine. Instantiate once before the loop, call .update(gesture) per frame.

class ModeMachine:
    """Owns the mode state and the debounced transition rules.

    Single instance lives for the lifetime of the app. The loop calls
    .update(gesture) once per frame; reads come via .current.
    """

    # --- Rules (class attrs — same for every instance, no need to copy) ---

    # Mode labels. Strings (not enums) so they compose easily into TRANSITIONS
    # keys and render directly into the on-screen mode indicator (Stage 9g).
    IDLE = "idle"   # default: no shape-creation gestures; selection + export work.
    DRAW = "draw"   # actively creating a shape; preview tracks hand position.

    # Allowed transitions: (current_mode, trigger_gesture) → target_mode.
    # Adding a new mode means adding rows here — no other dispatcher needed.
    TRANSITIONS = {
        (IDLE, "prayer_hands"): DRAW,   # prayer hands held in Idle → enter Draw
        (DRAW, "fist"):         IDLE,   # single fist held in Draw → exit to Idle
    }

    # How many consecutive frames the trigger gesture must hold to commit the flip.
    # 7 frames ≈ 230ms at 30fps — long enough to filter noise, short enough to feel snappy.
    DEBOUNCE_FRAMES = 7

    def __init__(self):
        # --- Runtime state (instance attrs — mutated each frame) ---

        # The mode we're actually in right now. Mutated when a debounced
        # transition completes, or by force_to() when the loop needs an
        # instant flip (e.g. commit auto-exits Draw).
        self.current = self.IDLE
        # The target we'd flip to if the trigger keeps holding.
        # None = no in-flight transition.
        self.pending = None
        # Consecutive frames the pending trigger has held. Resets to 0
        # (strict) on any miss frame — see the strict-reset note in update().
        self.pending_count = 0

    def update(self, gesture):
        """Per-frame state advancer. Driven by the frame's gesture label.

        Strict reset on miss: one frame without the trigger gesture zeros the
        counter. We chose this over 'tolerate N missed frames' because a wrong
        mode change costs more than redoing the hold.
        """
        # gets the value from the key-value of the Transitions dict
        target = self.TRANSITIONS.get((self.current, gesture)) 
        
        if target is not None:
            if self.pending == target:
                # Same target as last frame — continue the hold.
                self.pending_count += 1
                if self.pending_count >= self.DEBOUNCE_FRAMES:
                    # Hold completed: commit the flip.
                    self.current = target
                    # Clear the pending tracking — we're in the target mode now,
                    # so any future transition starts fresh from this state.
                    self.pending = None
                    self.pending_count = 0
            else:
                # New (or different) target this frame. Start at 1, not 0:
                # this frame already counts as the first observation.
                self.pending = target
                self.pending_count = 1
        else:
            # No transition matches this frame — strict reset (one miss
            # zeros it). Wrong mode change costs more than redoing the hold.
            self.pending = None
            self.pending_count = 0

    def force_to(self, target):
        """Instantly set current mode and clear any in-flight transition.

        Used by Stage 7's commit branch — two-fists auto-exits Draw to Idle
        without waiting for any debounce, AND any half-counted prayer_hands
        transition needs to be killed since the mode just changed instantly.
        """
        self.current = target
        self.pending = None
        self.pending_count = 0


# Single instance for the lifetime of the app.
mode = ModeMachine()


# ----------------------------------------------------------------------------
# Shape model
# ----------------------------------------------------------------------------
#
# Each committed shape is a dict with at least "type"; coordinates are in
# pixel space (already scaled by frame w/h). Preview is the same shape dict,
# rendered with alpha until commit.

# The diagram itself: a flat list. Order matters — later shapes draw on top,
# and hit-testing iterates back-to-front so the topmost shape wins selection.
shapes = []
# Ghost shape rendered while a shape gesture is held. Becomes a real entry
# in `shapes` when two-fists commits. Cleared when leaving Draw.
previewing_shape = None

# Selection / grab state — reactive (no Select mode), only active in Idle.
# Two-stage interaction: peace_open hovers → highlight; peace_closed grabs.
# All of this is encapsulated in the Selection class defined below the loop's
# imports. One instance lives for the app's lifetime; the loop calls
# selection.update(...) once per frame. See class def for the per-field whys.

# Shape rendering constants (SHAPE_COLOR, alphas, fixed sizes, arrow dims,
# DELETE_THRESHOLD) all live in shape.py — they belong with the geometry and
# drawing code that uses them. Only the pure-render constants for Stage 9h's
# warning border stay here; they're a driver concern, not a shape concern.
DELETE_BORDER_COLOR = (255, 20, 147)   # neon pink — visually distinct from anything else on screen.
DELETE_BORDER_WIDTH = 12               # thick enough to be unmistakable as a warning.

# Export: one fist held alone in Idle for EXPORT_HOLD_SECS triggers save.
EXPORT_HOLD_SECS = 3.0                 # long enough to be deliberate, short enough to feel responsive.
EXPORT_CIRCLE_RADIUS = 50              # radius of the on-screen progress indicator.
EXPORT_CIRCLE_WIDTH = 5                # thickness of the filling arc.
EXPORT_TRACK_COLOR = (100, 100, 100)   # gray background ring (the unfilled track).
EXPORT_FILL_COLOR = (0, 200, 0)        # green arc that fills clockwise as the hold progresses.

# Timestamp (seconds since epoch) when the current export hold began.
# None means no hold is in progress. Reset to None on any miss frame.
export_fist_start = None

# Note: the shared `preview_surface` scratchpad for translucent draws lives
# in shape.py now. The driver calls shape.init_preview_surface(w, h) on the
# first frame, once the camera resolution is known.

# ----------------------------------------------------------------------------
# HandLandmarker
# ----------------------------------------------------------------------------

# VIDEO mode = sync, tracks across frames using timestamps. The three confidence
# thresholds control how strict detection/presence/tracking are; 0.5 is default.
options = mp_vision.HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=mp_vision.RunningMode.VIDEO,
    num_hands=2,                            # max the model supports; we never use more.
    min_hand_detection_confidence=0.5,      # threshold for spotting a new hand.
    min_hand_presence_confidence=0.5,       # threshold for "this hand is still here."
    min_tracking_confidence=0.5,            # threshold for cross-frame tracking quality.
)
# Build the landmarker once and reuse — recreation per frame would be costly.
landmarker = mp_vision.HandLandmarker.create_from_options(options)


# ----------------------------------------------------------------------------
# Webcam
# ----------------------------------------------------------------------------

# Index 0 = system default camera. cv2 opens it as a video stream we can poll.
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    # macOS gotcha: this also fires when camera permission is denied silently.
    raise RuntimeError(
        "Could not open the webcam. Check System Settings → Privacy & Security → Camera."
    )


# ----------------------------------------------------------------------------
# Render loop — state + per-stage functions + the loop itself
# ----------------------------------------------------------------------------

# Anchor for monotonic ms timestamps — read by detect_hands(). Setting this
# right before the loop means timestamps start near 0 for the first frame.
loop_start = time.time()

# --- Profiling (opt-in via `python app.py --profile`) ---
# Off by default so normal runs pay nothing. When on, we accumulate wall-clock
# time for the whole loop body and for MediaPipe inference alone, then print a
# rolling average to the terminal every PROFILE_EVERY frames. detect_hands()
# always writes to _prof; the accumulators just stay unread when PROFILE is off.
PROFILE = "--profile" in sys.argv
PROFILE_EVERY = 30                       # frames between terminal reports (~1/sec at 30fps).
_prof = {"frames": 0, "loop_s": 0.0, "detect_s": 0.0}

# Pygame window and font. Created inside ensure_window() on the first iteration
# because we need the camera's resolution to size the window correctly, and we
# only learn that resolution from the first successful frame read.
screen = None
font = None


# ----------------------------------------------------------------------------
# Per-frame stages. Loop body below is a sequence of calls into these.
# ----------------------------------------------------------------------------
#
# Functions that mutate module state declare `global`; functions that only
# read state rely on Python's normal module-scope lookup. Each function name
# matches one (or a pair of) stages we've been calling out in comments.

def handle_events():
    """Stage 1: pump pygame events. Returns False if the user quit."""
    # Draining the queue every frame is what tells the OS the app is alive —
    # without this the window freezes (same reason cv2 has cv2.waitKey).
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            # User clicked the window's close button.
            return False
        if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
            # User pressed 'q' — alternate quit path (no window control needed).
            return False
    return True


def read_frame():
    """Stage 2: pull one frame from the webcam, mirror, convert color.

    Returns (frame_bgr, frame_rgb) or None on transient read failure.
    """
    # cap.read() returns (success, frame). frame is a HxWx3 numpy array of pixels.
    success, frame_bgr = cap.read()
    if not success:
        # Transient read failure (camera busy, glitch). Caller will skip this
        # frame and try again next iteration rather than crashing the loop.
        return None
    # Mirror so the user's right hand appears on the right side of the window —
    # matches the "mirror" mental model people expect from a webcam feed.
    frame_bgr = cv2.flip(frame_bgr, 1)
    # OpenCV gives BGR; MediaPipe wants RGB. pygame also wants RGB, so we
    # use frame_rgb for both the detector input and the background blit.
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return frame_bgr, frame_rgb


def detect_hands(frame_rgb):
    """Stage 3: run MediaPipe on the RGB frame.

    Returns the result with .hand_landmarks and .handedness parallel lists.
    """
    # Tasks API takes an mp.Image wrapper, not a raw numpy array.
    # SRGB is the standard color space; matches what the webcam produces.
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    # VIDEO mode demands monotonically-increasing ms timestamps so it can
    # track hands across frames. Anchored at loop_start so values start near 0.
    timestamp_ms = int((time.time() - loop_start) * 1000)
    # Synchronous call. `result` exposes two parallel lists:
    #   result.hand_landmarks[i] = list of 21 landmark points for hand i
    #   result.handedness[i]     = "Left"/"Right" predictions for hand i
    # perf_counter (monotonic, high-res) brackets ONLY the inference call, so
    # this measures MediaPipe cost in isolation from capture and render.
    t = time.perf_counter()
    result = landmarker.detect_for_video(mp_image, timestamp_ms)
    _prof["detect_s"] += time.perf_counter() - t
    return result


def ensure_window(w, h):
    """Stage 4: idempotent first-frame setup. Creates window/font/preview surface."""
    global screen, font
    if screen is not None:
        return
    # First-frame setup: deferred until now because we didn't know
    # the camera's resolution at module load time.
    screen = pygame.display.set_mode((w, h))
    pygame.display.set_caption("DiHand")
    # 32pt is the default size used for both mode label and per-hand text.
    font = pygame.font.SysFont(None, 32)
    # Allocate shape.py's alpha scratchpad now that we know the size.
    # All translucent draws (committed shapes, previews) blit through it.
    init_preview_surface(w, h)


def classify_gestures(hands):
    """Stage 5: per-hand labels + frame's combined gesture. Returns (labels, gesture)."""
    # Cache per-hand labels once. Used by both the gesture pipeline AND the
    # selection/grab block (selection.update reads hand_labels) — avoids
    # redundant classify calls.
    hand_labels = [classify(hand) for hand in hands]

    # One gesture-of-the-frame label drives mode + shape decisions below.
    # The hand-count branching exists so two separated open palms don't
    # fall back to single-hand 'open_palm' and accidentally exit modes.
    if len(hands) == 2:
        # Two-hand combo classifier. Returns None if no known combo matches —
        # that's intentional (no fallback to single-hand label).
        gesture = classify_two_hand(hands)
    elif len(hands) == 1:
        # Exactly one hand visible — its individual label is the frame's gesture.
        gesture = hand_labels[0]
    else:
        # Zero hands: no gesture this frame.
        gesture = None
    return hand_labels, gesture


def update_preview(gesture, hands, w, h, hands_handedness):
    """Stage 7: build/clear the active preview, commit on two-fists.

    Mutates `previewing_shape`. On commit also appends to `shapes` and
    instantly flips mode to IDLE via mode.force_to().
    """
    global previewing_shape
    # Shape preview & commit. Only runs meaningfully in Draw mode.
    if mode.current != mode.DRAW:
        # Outside Draw, a stale preview shouldn't linger — clear it.
        previewing_shape = None
    elif gesture in GESTURE_BUILDERS:
        # Shape gesture detected. Rebuild preview from this frame's landmarks
        # so it tracks hand position in real time (shape is fixed-size).
        # syntax is it looks up the function and passes (hands, w, h, hands_handedness)
        previewing_shape = GESTURE_BUILDERS[gesture](hands, w, h, hands_handedness)
    elif gesture == "two_fists" and previewing_shape is not None:
        # Commit: instant, no debounce. The preview becomes a real shape.
        shapes.append(previewing_shape)
        previewing_shape = None
        # Commit auto-exits Draw — one shape per Draw session by design.
        # force_to() also clears the in-flight debounce state, since the
        # mode just changed instantly and any half-counted prayer_hands
        # transition is now stale.
        mode.force_to(mode.IDLE)
    # No final `else` on purpose: if the gesture flickers off for a frame
    # mid-pose, the previous frame's preview persists. Less jittery UX.


def update_export_timer(gesture, hands, w, h):
    """Stage 8: track the one-fist hold; fire export when it completes."""
    global export_fist_start
    # Export hold timer. All three conditions must hold every frame —
    # any failure (mode change, second hand appears, fist lost) cancels.
    if mode.current == mode.IDLE and gesture == "fist" and len(hands) == 1:
        if export_fist_start is None:
            # First valid frame of the hold — start the timer now.
            export_fist_start = time.time()
        elif time.time() - export_fist_start >= EXPORT_HOLD_SECS:
            # 3 seconds elapsed — fire export and clear so we don't re-fire.
            export_diagram(shapes, w, h)
            export_fist_start = None
    else:
        # Any miss instantly cancels the hold — stricter than GRAB_TOLERANCE
        # because accidental export is worse than redoing a 3-second hold.
        export_fist_start = None


def draw_background(frame_rgb, w, h):
    """Stage 9a: blit the webcam frame as the canvas background.

    Also acts as a "clear" — anything drawn on screen last frame gets overwritten.
    """
    frame_surface = pygame.image.frombuffer(frame_rgb.tobytes(), (w, h), "RGB")
    screen.blit(frame_surface, (0, 0))


def draw_skeleton(hands, hand_labels, w, h):
    """Stages 9b + 9c: hand skeleton overlay + per-hand debug labels.

    Skeleton color tints with mode: orange = Idle, green = Draw — at-a-glance
    mode awareness without reading the top-left text.
    """
    skeleton_color = (0, 255, 0) if mode.current == mode.DRAW else (255, 200, 0)
    for hi, hand in enumerate(hands):
        # Bones first, then dots, so dots sit on top.
        for start_idx, end_idx in HAND_CONNECTIONS:
            # Landmark .x/.y are normalized [0,1] — multiply by w/h for pixels.
            start_pt = (int(hand[start_idx].x * w), int(hand[start_idx].y * h))
            end_pt   = (int(hand[end_idx].x * w),   int(hand[end_idx].y * h))
            pygame.draw.line(screen, skeleton_color, start_pt, end_pt, 2)
        for lm in hand:
            pygame.draw.circle(screen, skeleton_color, (int(lm.x * w), int(lm.y * h)), 4)

        # === Stage 9c: per-hand label (debug aid) ===
        # Shows the classify() result under the wrist so you can see what
        # gesture the system thinks each hand is doing in real time.
        wrist_x = int(hand[WRIST].x * w)
        wrist_y = int(hand[WRIST].y * h)
        label_surface = font.render(hand_labels[hi], True, (255, 0, 0))
        # Offset (-50, +20) keeps the label below + slightly left of the wrist.
        screen.blit(label_surface, (wrist_x - 50, wrist_y + 20))


def draw_shapes_and_preview():
    """Stages 9e + 9f: committed shapes (opaque outline + translucent fill)
    then the active preview (lower opacity).

    Both paint the same shape model with different alphas, hence one function.
    """
    # === Stage 9e: render committed shapes ===
    # Drawn after skeleton so the diagram sits on top of the user's hand.
    # Loop var is `s` (not `shape`) to keep the habit — module-level
    # `for shape in ...` would leak into the enclosing scope (Python
    # doesn't scope for-vars) and risk shadowing imports.
    for i, s in enumerate(shapes):
        # Either grabbed or highlighted → render in HIGHLIGHT_COLOR.
        is_highlighted = (i == selection.grabbed_index) or (i == selection.highlighted_index)
        draw_shape(screen, s, highlighted=is_highlighted)
    # === Stage 9f: render active preview ===
    # Drawn at lower opacity so it visually reads as "not yet real."
    if previewing_shape is not None:
        draw_shape(screen, previewing_shape, fill_alpha=PREVIEW_ALPHA)


def draw_overlays(in_delete_zone, w, h):
    """Stages 9g + 9h + 9i: mode indicator, delete-zone warning border,
    export progress arc.

    All UI signals that sit on top of the diagram. Drawn last so they're
    unmissable.
    """
    # === Stage 9g: mode indicator text ===
    # Top-left corner. White on whatever's behind = always readable.
    mode_surface = font.render(f"MODE: {mode.current.upper()}", True, (255, 255, 255))
    screen.blit(mode_surface, (20, 20))

    # === Stage 9h: delete-zone warning ===
    # Drawn after shapes so the pink border sits on top of everything else,
    # making the warning unmissable when a drag enters the zone.
    if in_delete_zone:
        pygame.draw.rect(screen, DELETE_BORDER_COLOR, (0, 0, w, h), DELETE_BORDER_WIDTH)

    # === Stage 9i: export progress circle ===
    # Only shown while a one-fist hold is in progress. Gray track behind,
    # green arc filling clockwise from 12 o'clock as the hold approaches 3s.
    if export_fist_start is not None:
        elapsed = time.time() - export_fist_start
        # Clamp at 1.0 in case the hold tips slightly past EXPORT_HOLD_SECS
        # before the export call clears the timer next frame.
        progress = min(elapsed / EXPORT_HOLD_SECS, 1.0)
        # Centered on the screen.
        cx, cy = w // 2, h // 2
        # Bounding box used by pygame.draw.arc.
        rect = pygame.Rect(cx - EXPORT_CIRCLE_RADIUS, cy - EXPORT_CIRCLE_RADIUS,
                           EXPORT_CIRCLE_RADIUS * 2, EXPORT_CIRCLE_RADIUS * 2)
        # Track ring (gray, 2px outline).
        pygame.draw.circle(screen, EXPORT_TRACK_COLOR, (cx, cy), EXPORT_CIRCLE_RADIUS, 2)
        if progress > 0:
            # Pygame angles: 0 = right, π/2 = top, grow counterclockwise.
            # Start at top and sweep back (counterclockwise in angle space)
            # by 2π·progress — which renders as a clockwise fill on screen
            # because the y-axis flips.
            start_angle = math.pi / 2 - 2 * math.pi * progress
            stop_angle = math.pi / 2
            pygame.draw.arc(screen, EXPORT_FILL_COLOR, rect, start_angle, stop_angle, EXPORT_CIRCLE_WIDTH)


def cleanup():
    """Finally-block cleanup: release the three independent resources we hold."""
    cap.release()       # release the camera so other apps can use it.
    pygame.quit()       # tear down pygame subsystems (window, fonts, etc.).
    landmarker.close()  # free the MediaPipe model's memory.


# ----------------------------------------------------------------------------
# The loop. Each frame is the same sequence: input → think → render → flip.
# ----------------------------------------------------------------------------

try:
    while True:
        # Bracket the whole body so the fps figure reflects everything the loop
        # does per frame (capture + detect + classify + render), not just one stage.
        _t0 = time.perf_counter()
        if not handle_events():
            # User quit — break immediately so we don't render half a frame.
            break
        frame = read_frame()
        if frame is None:
            # Transient read failure — skip this frame and try again.
            continue
        frame_bgr, frame_rgb = frame
        result = detect_hands(frame_rgb)
        # Camera resolution from the actual frame. Downstream stages need w/h
        # to scale normalized landmark coords into pixels.
        h, w = frame_bgr.shape[:2]
        ensure_window(w, h)

        # `or []` handles the no-hands case — MediaPipe returns None when
        # it finds nothing, which would break `len()` and iteration downstream.
        hands = result.hand_landmarks or []
        hands_handedness = result.handedness or []
        hand_labels, gesture = classify_gestures(hands)

        # === think ===
        mode.update(gesture)
        update_preview(gesture, hands, w, h, hands_handedness)
        update_export_timer(gesture, hands, w, h)

        # === render ===
        draw_background(frame_rgb, w, h)
        draw_skeleton(hands, hand_labels, w, h)
        # Stage 9d: selection also mutates `shapes` (drag/delete) — has to
        # run before draw_shapes_and_preview so the latter sees updated positions.
        in_delete_zone = selection.update(hands, hand_labels, shapes, mode, w, h)
        draw_shapes_and_preview()
        draw_overlays(in_delete_zone, w, h)

        # Stage 10: present the assembled frame — the moment the user sees it.
        pygame.display.flip()

        # Profiling report. Accumulate this frame's body time, then every
        # PROFILE_EVERY frames print rolling averages and reset — a fresh window
        # each report so the numbers track the *current* run, not the lifetime average.
        if PROFILE:
            _prof["loop_s"] += time.perf_counter() - _t0
            _prof["frames"] += 1
            if _prof["frames"] >= PROFILE_EVERY:
                n = _prof["frames"]
                fps = n / _prof["loop_s"] if _prof["loop_s"] else 0.0
                detect_ms = _prof["detect_s"] / n * 1000
                frame_ms = _prof["loop_s"] / n * 1000
                print(f"[profile] {fps:5.1f} fps | frame {frame_ms:5.1f} ms | detect {detect_ms:5.1f} ms")
                _prof = {"frames": 0, "loop_s": 0.0, "detect_s": 0.0}
finally:
    cleanup()
