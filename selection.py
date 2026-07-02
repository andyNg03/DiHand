"""
selection.py — reactive hover + grab/drag/delete state machine.

Owns the selection/grab state and the per-frame logic that updates it.
Single instance for the lifetime of the app. The driver calls
selection.update(...) once per frame; in_delete_zone is returned for
the warning border. Mutates the shared `shapes` list when a grabbed
shape is dragged or deleted.

Leaf module: depends only on `shape` (geometry helpers) and `gestures`
(landmark indices). `mode` is duck-typed — Selection reads `.current`
and `.DRAW` off whatever is passed in, so this file doesn't import
ModeMachine and isn't coupled to where it lives.
"""

from gestures import INDEX_TIP, MIDDLE_TIP
from shape import shape_contains, translate_shape, is_out_of_bounds


class Selection:
    """Hover (peace_open) + grab/drag (peace_closed) state, suppressed in Draw."""

    # How many consecutive frames without peace_closed before a grab releases.
    # During fast drags MediaPipe can lose the hand for a frame or two; tolerance
    # prevents the grab from dropping the moment that happens. 10 ≈ 100ms at 30fps.
    GRAB_TOLERANCE = 10

    def __init__(self):
        # --- Runtime state (instance attrs — mutated each frame) ---

        # Index in `shapes` of the shape currently under the V-center, or None.
        # Updated by hover scans (peace_open). Persists across the open→closed
        # transition so the user doesn't lose selection while pinching fingers.
        self.highlighted_index = None
        # Index in `shapes` of the shape currently being dragged, or None.
        # Set when peace_closed initiates a grab; cleared on release or delete.
        self.grabbed_index = None
        # V-center x/y from the last frame — for computing this frame's drag delta.
        # Without these, the shape would snap to the hand on grab; the delta
        # approach makes it follow the hand from wherever it was when grabbed.
        self.last_grab_vx = 0.0
        self.last_grab_vy = 0.0
        # Consecutive frames without peace_closed while a grab is active.
        # When this reaches GRAB_TOLERANCE, the grab actually releases.
        self.grab_miss_count = 0

    @staticmethod
    def _v_center(hand, w, h):
        """Midpoint of INDEX_TIP and MIDDLE_TIP in pixel coords — anchor for hover/drag."""
        # The natural "pointing" anchor for a peace sign — between the two
        # extended fingers. Used identically by hover scan and drag delta.
        vx = (hand[INDEX_TIP].x + hand[MIDDLE_TIP].x) / 2 * w
        vy = (hand[INDEX_TIP].y + hand[MIDDLE_TIP].y) / 2 * h
        return vx, vy

    def update(self, hands, hand_labels, shapes, mode, w, h):
        """Per-frame dispatcher. Returns in_delete_zone for Stage 9h.

        Three mutually-exclusive branches based on (mode.current, grabbed_index):
        suppress (in Draw), drag-or-release (grab active), hover-or-grab (idle).
        """
        if mode.current == mode.DRAW:
            # Selection is suppressed in Draw so peace gestures don't accidentally
            # move committed shapes while the user is creating new ones.
            return self._suppress()
        if self.grabbed_index is not None:
            # An active grab exists — translate this frame or count toward release.
            return self._drag_or_release(hands, hand_labels, shapes, w, h)
        # No active grab — peace_open hovers, peace_closed initiates grab.
        return self._hover_or_grab(hands, hand_labels, shapes, w, h)

    def _suppress(self):
        # Drop any active grab and clear the highlight so they don't surprise
        # the user when they re-enter Idle later.
        if self.grabbed_index is not None:
            self.grabbed_index = None
        self.highlighted_index = None
        # No delete-zone check possible without a grab; always False here.
        return False

    def _drag_or_release(self, hands, hand_labels, shapes, w, h):
        # While dragging, only look for the peace_closed hand to continue or
        # release. Skip the highlight scan — the highlight is fixed for the
        # duration of the grab (it's the same as grabbed_index visually).
        any_closed = False
        in_delete_zone = False
        for hi, hand in enumerate(hands):
            if hand_labels[hi] == "peace_closed":
                # Hand still in grab pose — compute drag delta and translate the shape.
                any_closed = True
                vx, vy = self._v_center(hand, w, h)
                # Delta vs last frame — the shape shifts by exactly this much,
                # so it follows the hand without snapping to it on grab.
                dx = vx - self.last_grab_vx
                dy = vy - self.last_grab_vy
                shapes[self.grabbed_index] = translate_shape(shapes[self.grabbed_index], dx, dy)
                self.last_grab_vx, self.last_grab_vy = vx, vy
                # Check delete-zone proximity for the warning border + release logic.
                in_delete_zone = is_out_of_bounds(shapes[self.grabbed_index], w, h)
                # Reset miss-count on every successful frame.
                self.grab_miss_count = 0
                # Only one hand drives the drag per frame, even if both are closed.
                break
        if not any_closed:
            # No peace_closed this frame — count toward eventual release.
            # GRAB_TOLERANCE lets transient MediaPipe flickers pass without dropping.
            self.grab_miss_count += 1
            if self.grab_miss_count >= self.GRAB_TOLERANCE:
                # Sustained miss — release the grab.
                if is_out_of_bounds(shapes[self.grabbed_index], w, h):
                    # Released inside the delete zone → actually delete the shape.
                    shapes.pop(self.grabbed_index)
                self.grabbed_index = None
                self.grab_miss_count = 0
        return in_delete_zone

    def _hover_or_grab(self, hands, hand_labels, shapes, w, h):
        # No active grab — peace_open highlights, peace_closed grabs (only if
        # something is already highlighted, enforcing the deliberate two-step).
        new_highlight = None   # which shape (if any) gets the highlight this frame.
        any_closed = False     # did we see a peace_closed hand this frame?
        for hi, hand in enumerate(hands):
            label = hand_labels[hi]
            if label == "peace_open":
                # Hover: hit-test from front (last) to back (first) so the
                # topmost shape wins — same z-order users see on screen.
                vx, vy = self._v_center(hand, w, h)
                for i in range(len(shapes) - 1, -1, -1):
                    if shape_contains(shapes[i], (vx, vy)):
                        new_highlight = i
                        break
            elif label == "peace_closed" and self.highlighted_index is not None:
                # Grab is only allowed if something is already highlighted —
                # forces the deliberate two-step (hover, then grab).
                vx, vy = self._v_center(hand, w, h)
                self.grabbed_index = self.highlighted_index
                self.last_grab_vx, self.last_grab_vy = vx, vy
                any_closed = True
                # Stop scanning — we have a grab; ignore other hands this frame.
                break

        if not any_closed:
            # Highlight bookkeeping. We want the highlight to persist across
            # the open→closed transition (so the user doesn't lose selection
            # mid-pinch), but to clear when no peace pose is visible at all.
            has_peace = any(l in ("peace_open", "peace_closed") for l in hand_labels)
            if new_highlight is not None:
                # Fresh peace_open hover this frame — update the highlight.
                self.highlighted_index = new_highlight
            elif not has_peace:
                # No peace hands at all — clear the highlight.
                self.highlighted_index = None
            # else: keep the existing highlighted_index (transition period).
        # _hover_or_grab can't be in the delete zone — that's a drag-only state.
        return False
