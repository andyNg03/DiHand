"""
gestures.py — hand-gesture rules over MediaPipe's 21 landmarks.

Single-hand classifier design:
  - 3D distance (z included) — 2D foreshortens when the palm tilts away.
  - Thumb excluded from the open/closed vote — bends sideways across the palm.
  - Asymmetric thresholds — open hand needs slack for tilt; fist doesn't.
  - Strict 4/4 vote — prefer "unknown" over a confident wrong label.

Two-hand gestures (returned by `classify_two_hand`):
  - prayer_hands — both open_palm, vertical, wrists close, fingertips aligned.
  - two_fists   — both fist. Used to commit a previewed shape.
  - l_frames    — both in L-pose (index + thumb extended, three others curled).
"""

import math

# Landmark indices we reference by name.
WRIST = 0
THUMB_TIP = 4
INDEX_MCP, INDEX_TIP = 5, 8
MIDDLE_MCP, MIDDLE_TIP = 9, 12
RING_MCP, RING_TIP = 13, 16
PINKY_MCP, PINKY_TIP = 17, 20

NON_THUMB_FINGERS = [
    (INDEX_MCP, INDEX_TIP),
    (MIDDLE_MCP, MIDDLE_TIP),
    (RING_MCP, RING_TIP),
    (PINKY_MCP, PINKY_TIP),
]

# ratio = dist(wrist, tip) / dist(wrist, mcp). Extended ≈ 2.0, curled ≈ 1.0.
EXTENDED_RATIO = 1.5
CURLED_RATIO = 0.9

# Wrist-to-wrist distance for prayer hands, as multiple of palm width. Can't
# be tighter than ~1.0: touching palms occlude each other and collapse to one.
PRAYER_HANDS_RATIO = 3.0

# Corresponding-fingertip proximity for prayer hands. Both ends close = the
# "tent" pose, not just casually close hands.
FINGERTIP_PROXIMITY_RATIO = 1.5

# Thumb extension proxy: dist(THUMB_TIP, MIDDLE_MCP) / palm_width.
# Extended thumb is ~1.0+, tucked (curled across palm) thumb is ~0.3-0.5.
THUMB_EXTENDED_RATIO = 0.8

# Triangle frame: dist(INDEX_TIPs) / dist(WRISTs). Below threshold = indices
# converge enough to form an apex. Square's natural ratio is ~0.80 — keep
# buffer below that to avoid collision.
TRIANGLE_CONVERGENCE_RATIO = 0.7

# Circle frame: max non-thumb fingertip distance from each hand's fingertip
# centroid, as fraction of palm width. Small = tight "claw" pose.
CIRCLE_CLUSTER_RATIO = 0.5

# Circle-specific finger ratio band. Tighter than the global unsure band
# (CURLED_RATIO..EXTENDED_RATIO) to reject fingers that are just passing
# through on the way to a fist.
CIRCLE_RATIO_LOW = 1.05
CIRCLE_RATIO_HIGH = 1.4

# Peace sign: dist(INDEX_TIP, MIDDLE_TIP) / palm_width must exceed this for the
# "open" variant. Below it the fingers are touching/closed — handled in 7b.
PEACE_OPEN_RATIO = 0.45


def _dist(a, b):
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)


def _finger_state(hand, mcp_idx, tip_idx):
    """'extended', 'curled', or 'unsure' for a single non-thumb finger."""
    ratio = _dist(hand[WRIST], hand[tip_idx]) / _dist(hand[WRIST], hand[mcp_idx])
    if ratio > EXTENDED_RATIO:
        return "extended"
    if ratio < CURLED_RATIO:
        return "curled"
    return "unsure"


def _thumb_extended(hand):
    """True if the thumb is extended away from the palm."""
    palm_width = _dist(hand[INDEX_MCP], hand[PINKY_MCP])
    return _dist(hand[THUMB_TIP], hand[MIDDLE_MCP]) > palm_width * THUMB_EXTENDED_RATIO


def classify(hand):
    """Return 'open_palm', 'fist', 'peace_open', 'peace_closed', or 'unknown'."""
    states = [_finger_state(hand, mcp, tip) for mcp, tip in NON_THUMB_FINGERS]
    extended = sum(1 for s in states if s == "extended")
    curled = sum(1 for s in states if s == "curled")
    if extended == 4:
        return "open_palm"
    if curled == 4:
        return "fist"
    # Peace: INDEX + MIDDLE extended, RING + PINKY curled, thumb ignored.
    # Open (V-spread) vs closed (fingers touching) split at PEACE_OPEN_RATIO.
    if (states[0] == "extended" and states[1] == "extended"
            and states[2] == "curled" and states[3] == "curled"):
        palm_width = _dist(hand[INDEX_MCP], hand[PINKY_MCP])
        if _dist(hand[INDEX_TIP], hand[MIDDLE_TIP]) > palm_width * PEACE_OPEN_RATIO:
            return "peace_open"
        return "peace_closed"
    return "unknown"


def _is_l_frame(hand):
    """Open palm (all 4 non-thumb extended) + thumb extended + index pointing up.

    Strict L-pose (only index + thumb extended) is intentionally NOT accepted —
    the user wants square to require a full open hand with thumb out, so the
    classic L-shape doesn't accidentally trigger it.
    """
    if classify(hand) != "open_palm":
        return False
    if not _thumb_extended(hand):
        return False
    return hand[INDEX_TIP].y < hand[WRIST].y


def _is_triangle_frame(a, b):
    """Both hands: index + middle + ring extended, pinky NOT extended; INDEX_TIPs converging.

    Pinky check prevents open-palm (all 4 extended) from leaking into diamond
    when the thumb drops during the square gesture.
    """
    for hand in (a, b):
        for mcp_idx, tip_idx in [(INDEX_MCP, INDEX_TIP), (MIDDLE_MCP, MIDDLE_TIP), (RING_MCP, RING_TIP)]:
            if _finger_state(hand, mcp_idx, tip_idx) != "extended":
                return False
        if _finger_state(hand, PINKY_MCP, PINKY_TIP) == "extended":
            return False
    wrist_dist = _dist(a[WRIST], b[WRIST])
    if wrist_dist == 0:
        return False
    return _dist(a[INDEX_TIP], b[INDEX_TIP]) / wrist_dist < TRIANGLE_CONVERGENCE_RATIO


def _is_circle_frame(hand):
    """The 'claw': all 4 non-thumb fingers in a tight mid-curl band,
    fingertips clustered. Uses CIRCLE_RATIO_LOW/HIGH instead of the global
    unsure band so fingers just passing through on the way to a fist don't fire."""
    for mcp_idx, tip_idx in NON_THUMB_FINGERS:
        ratio = _dist(hand[WRIST], hand[tip_idx]) / _dist(hand[WRIST], hand[mcp_idx])
        if ratio < CIRCLE_RATIO_LOW or ratio > CIRCLE_RATIO_HIGH:
            return False
    tips = [hand[INDEX_TIP], hand[MIDDLE_TIP], hand[RING_TIP], hand[PINKY_TIP]]
    cx = sum(t.x for t in tips) / 4
    cy = sum(t.y for t in tips) / 4
    cz = sum(t.z for t in tips) / 4
    max_spread = max(
        math.sqrt((t.x - cx) ** 2 + (t.y - cy) ** 2 + (t.z - cz) ** 2) for t in tips
    )
    palm_width = _dist(hand[INDEX_MCP], hand[PINKY_MCP])
    return max_spread < palm_width * CIRCLE_CLUSTER_RATIO


def _is_pointing(hand):
    """Index extended, the other three fingers genuinely curled. Thumb free."""
    if _finger_state(hand, INDEX_MCP, INDEX_TIP) != "extended":
        return False
    for mcp_idx, tip_idx in [(MIDDLE_MCP, MIDDLE_TIP), (RING_MCP, RING_TIP), (PINKY_MCP, PINKY_TIP)]:
        if _finger_state(hand, mcp_idx, tip_idx) != "curled":
            return False
    return True


def _is_prayer_hands(a, b):
    if classify(a) != "open_palm" or classify(b) != "open_palm":
        return False
    # Both hands point up (middle fingertip above wrist on screen).
    if a[MIDDLE_TIP].y >= a[WRIST].y or b[MIDDLE_TIP].y >= b[WRIST].y:
        return False
    palm_width = _dist(a[INDEX_MCP], a[PINKY_MCP])
    if _dist(a[WRIST], b[WRIST]) >= palm_width * PRAYER_HANDS_RATIO:
        return False
    for _, tip_idx in NON_THUMB_FINGERS:
        if _dist(a[tip_idx], b[tip_idx]) >= palm_width * FINGERTIP_PROXIMITY_RATIO:
            return False
    return True


def classify_two_hand(hands):
    """Return a two-hand gesture label, or None if no two-hand gesture matches.

    Recognizes: 'prayer_hands', 'two_fists', 'l_frames', 'triangle_frame',
    'circle_frame', 'pointing'. Order matters — most specific checks come first
    so a looser later check can't shadow a stricter earlier one.
    """
    if len(hands) != 2:
        return None
    a, b = hands
    if _is_prayer_hands(a, b):
        return "prayer_hands"
    if classify(a) == "fist" and classify(b) == "fist":
        return "two_fists"
    if _is_l_frame(a) and _is_l_frame(b):
        return "l_frames"
    if _is_triangle_frame(a, b):
        return "triangle_frame"
    if _is_circle_frame(a) and _is_circle_frame(b):
        return "circle_frame"
    if _is_pointing(a) and _is_pointing(b):
        return "pointing"
    return None
