"""
shape.py — the Shape concept for DiHand.

Defines what a shape *is* (dict format), how to build one from hand landmarks,
how to draw it, hit-test it, translate it, and check whether it has drifted
out of the canvas bounds. The driver imports from here and works with the
returned dicts; this module never touches global app state besides the
shared `preview_surface` scratchpad declared below.

Shape dict formats:
  square:  {"type": "square",  "center": (cx, cy), "half": h}
  diamond: {"type": "diamond", "center": (cx, cy), "half_w": hw, "half_h": hh}
  circle:  {"type": "circle",  "center": (cx, cy), "radius": r}
  arrow:   {"type": "arrow",   "source": (sx, sy), "dest": (dx, dy)}
"""

import math
import pygame

from gestures import (
    WRIST, INDEX_MCP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP,
)


# ----------------------------------------------------------------------------
# Shape rendering constants
# ----------------------------------------------------------------------------
#
# Three opacity tiers: export = fully opaque, committed on the live canvas =
# translucent fill + opaque outline, preview = even lighter.

SHAPE_COLOR = (217, 34, 67)       # #D92243 — base color for all shape types.
HIGHLIGHT_COLOR = (0, 0, 255)     # blue — applied when hovered or grabbed.
CORNER_RADIUS = 8                 # rounded corners for squares + diamonds.
SHAPE_FILL_ALPHA = 60             # see-through enough that the hand shows through committed shapes.
SHAPE_OUTLINE_WIDTH = 2           # always drawn at full opacity, regardless of fill alpha.
PREVIEW_ALPHA = 50                # previews are even more transparent than committed shapes.

# Fixed shape sizes (Redesign phase). Position still tracks the hands; only
# size is locked, so shapes can't accidentally end up too big or too small.
FIXED_SQUARE_HALF = 55             # 110×110 square.
FIXED_DIAMOND_HALF = 55            # 110px diagonal diamond.
FIXED_CIRCLE_RADIUS = 50           # 100px diameter circle.

# Arrow: chunkier than the outline-only shapes — it has to read as a directional
# mark, not just another perimeter, so line and head are deliberately oversized.
ARROW_LINE_WIDTH = 4           # thicker than SHAPE_OUTLINE_WIDTH so the arrow stands out.
ARROW_HEAD_LEN = 32            # tip-to-base distance along the arrow axis.
ARROW_HEAD_HALF_WIDTH = 12     # perpendicular half-width at the base.
ARROW_MIN_LENGTH = 10          # skip rendering if source ≈ dest (avoid div-by-zero on the unit vector).
ARROW_HIT_TOLERANCE = 15       # arrows are line-thin, so we accept hits within 15px of the segment.

# Delete-by-drag: when a grabbed shape's center reaches within DELETE_THRESHOLD
# px of any screen edge, releasing the grab removes the shape. Lives here
# because is_out_of_bounds (below) uses it. The pink border colors stay in
# app.py — those are render-only concerns, not shape geometry.
DELETE_THRESHOLD = 60


# ----------------------------------------------------------------------------
# Shared alpha scratchpad
# ----------------------------------------------------------------------------
#
# Every translucent shape draw uses this surface: clear it, paint a fill (with
# alpha) and an outline (opaque) onto it, then blit onto the destination.
# Needed because pygame's main screen is opaque and can't accept alpha pixels.
# Allocated lazily on the first frame (we don't know the camera resolution at
# module load); the driver calls init_preview_surface(w, h) once.

preview_surface = None


def init_preview_surface(w, h):
    """Allocate the shared alpha scratchpad. Call once on the first frame."""
    global preview_surface
    # SRCALPHA = per-pixel alpha. Needed because translucent shape fills
    # have to be drawn here first, then blitted onto the destination screen.
    preview_surface = pygame.Surface((w, h), pygame.SRCALPHA)


# ----------------------------------------------------------------------------
# Shape builders — gesture + landmarks → shape dict (pixel coords)
# ----------------------------------------------------------------------------
#
# All four take the same signature (hands, frame_w, frame_h, handedness) so they
# can be dispatched uniformly by GESTURE_BUILDERS. Landmark .x/.y are normalized
# [0,1], so each builder multiplies by frame dimensions to get pixel coords.

def make_square(hands, frame_w, frame_h, handedness):
    """Fixed-size square centered between INDEX_MCPs."""
    # INDEX_MCP = the knuckle at the base of the index finger.
    # Using MCPs (not TIPs) gives a stable center even as fingers wiggle.
    p1 = (hands[0][INDEX_MCP].x * frame_w, hands[0][INDEX_MCP].y * frame_h)
    p2 = (hands[1][INDEX_MCP].x * frame_w, hands[1][INDEX_MCP].y * frame_h)
    # Center = midpoint between the two knuckles. Size is fixed (post-Redesign).
    cx, cy = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
    return {"type": "square", "center": (cx, cy), "half": FIXED_SQUARE_HALF}


def make_diamond(hands, frame_w, frame_h, handedness):
    """Fixed-size diamond centered between WRISTs and INDEX_TIPs."""
    # The triangle_frame gesture has both wrists low and indices high, so we
    # average all four points to get a center that sits in the visual middle
    # of the user's "framing" pose — not biased toward either hand or hand part.
    w1 = (hands[0][WRIST].x * frame_w, hands[0][WRIST].y * frame_h)
    w2 = (hands[1][WRIST].x * frame_w, hands[1][WRIST].y * frame_h)
    t1 = (hands[0][INDEX_TIP].x * frame_w, hands[0][INDEX_TIP].y * frame_h)
    t2 = (hands[1][INDEX_TIP].x * frame_w, hands[1][INDEX_TIP].y * frame_h)
    cx = (w1[0] + w2[0] + t1[0] + t2[0]) / 4
    cy = (w1[1] + w2[1] + t1[1] + t2[1]) / 4
    # Regular diamond: equal half_w and half_h = 45°-rotated square.
    return {"type": "diamond", "center": (cx, cy), "half_w": FIXED_DIAMOND_HALF, "half_h": FIXED_DIAMOND_HALF}


def make_arrow(hands, frame_w, frame_h, handedness):
    """Arrow: source = user's physical LEFT hand INDEX_TIP, dest = right hand's.

    We pre-mirror the frame, so MediaPipe sees a non-selfie image and labels
    handedness from the image's POV — its "Right" = user's physical left.
    Inverted from the docs' selfie-input assumption; verified by testing.
    """
    # handedness[i][0] = MediaPipe's top-confidence guess for hand i.
    # We use it (not screen position) so the arrow can point in any direction.
    if handedness[0][0].category_name == "Right":
        source_hand, dest_hand = hands[0], hands[1]
    else:
        source_hand, dest_hand = hands[1], hands[0]
    # INDEX_TIPs are the literal "pointing" points — where each finger is aimed.
    source = (source_hand[INDEX_TIP].x * frame_w, source_hand[INDEX_TIP].y * frame_h)
    dest = (dest_hand[INDEX_TIP].x * frame_w, dest_hand[INDEX_TIP].y * frame_h)
    return {"type": "arrow", "source": source, "dest": dest}


def make_circle(hands, frame_w, frame_h, handedness):
    """Fixed-size circle centered between per-hand fingertip centroids."""
    # Centroid of each hand's 4 non-thumb fingertips = the "center" of the claw pose.
    def centroid(hand):
        tips = [hand[INDEX_TIP], hand[MIDDLE_TIP], hand[RING_TIP], hand[PINKY_TIP]]
        return (
            sum(t.x for t in tips) * frame_w / 4,
            sum(t.y for t in tips) * frame_h / 4,
        )
    c1, c2 = centroid(hands[0]), centroid(hands[1])
    # Final center = midpoint between the two per-hand centroids = the imagined
    # center of the circle the user is forming with both hands.
    center = ((c1[0] + c2[0]) / 2, (c1[1] + c2[1]) / 2)
    return {"type": "circle", "center": center, "radius": FIXED_CIRCLE_RADIUS}


# ----------------------------------------------------------------------------
# Drawing primitives
# ----------------------------------------------------------------------------
#
# The translucent-shape draws all follow the same recipe: clear the alpha
# scratchpad → paint a fill (with alpha) and an outline (full opacity) onto it
# → blit the scratchpad onto the destination surface. The scratchpad is needed
# because pygame's main screen is opaque and can't directly accept alpha pixels.

def _rounded_polygon_points(vertices, radius, segments=5):
    """Quadratic Bezier at each corner to approximate rounded edges."""
    # For each vertex, replace the sharp corner with a smooth Bezier arc.
    # Result: a denser list of points that, when drawn as a polygon, looks rounded.
    n = len(vertices)
    points = []
    for i in range(n):
        # Three consecutive vertices: previous, current (the corner), next.
        prev = vertices[(i - 1) % n]
        curr = vertices[i]
        nxt = vertices[(i + 1) % n]
        # Vectors from the corner toward each neighbor — we'll walk inward
        # along these to find where the rounded arc should start and end.
        dx1, dy1 = prev[0] - curr[0], prev[1] - curr[1]
        dx2, dy2 = nxt[0] - curr[0], nxt[1] - curr[1]
        len1 = math.sqrt(dx1 * dx1 + dy1 * dy1)
        len2 = math.sqrt(dx2 * dx2 + dy2 * dy2)
        if len1 == 0 or len2 == 0:
            # Degenerate (zero-length edge) — just keep the corner sharp.
            points.append(curr)
            continue
        # Clamp the rounding radius so the arc never exceeds half an edge.
        # Otherwise corners on short edges would overshoot each other.
        r = min(radius, len1 / 2, len2 / 2)
        # Arc start (toward prev) and end (toward next), both `r` from the corner.
        p1 = (curr[0] + dx1 / len1 * r, curr[1] + dy1 / len1 * r)
        p2 = (curr[0] + dx2 / len2 * r, curr[1] + dy2 / len2 * r)
        # Sample `segments+1` points along the Bezier: B(t) = (1-t)²·p1 + 2t(1-t)·corner + t²·p2.
        # The corner itself is the control point that pulls the curve outward.
        for j in range(segments + 1):
            t = j / segments
            x = (1 - t) ** 2 * p1[0] + 2 * t * (1 - t) * curr[0] + t ** 2 * p2[0]
            y = (1 - t) ** 2 * p1[1] + 2 * t * (1 - t) * curr[1] + t ** 2 * p2[1]
            points.append((x, y))
    return points


def _draw_filled_rect(surface, rect, color, fill_alpha, border_radius):
    # Used by squares. Three steps: clear scratchpad, paint fill+outline with
    # per-pixel alpha, blit the scratchpad onto the destination.
    preview_surface.fill((0, 0, 0, 0))   # fully transparent reset.
    # width=0 means filled. color + (alpha,) extends the RGB tuple to RGBA.
    pygame.draw.rect(preview_surface, color + (fill_alpha,), rect, 0, border_radius=border_radius)
    # Outline always at full opacity (255), regardless of fill alpha.
    pygame.draw.rect(preview_surface, color + (255,), rect, SHAPE_OUTLINE_WIDTH, border_radius=border_radius)
    surface.blit(preview_surface, (0, 0))


def _draw_filled_polygon(surface, vertices, color, fill_alpha, corner_radius):
    # Used by diamonds. Same alpha recipe as the rect; only the geometry differs.
    # First subdivide the corners so the polygon's outline reads as rounded.
    pts = _rounded_polygon_points(vertices, corner_radius)
    preview_surface.fill((0, 0, 0, 0))
    pygame.draw.polygon(preview_surface, color + (fill_alpha,), pts, 0)
    pygame.draw.polygon(preview_surface, color + (255,), pts, SHAPE_OUTLINE_WIDTH)
    surface.blit(preview_surface, (0, 0))


def _draw_arrow(surface, source, dest, color, fill_alpha):
    """Line from source to dest, filled triangle head with tip at dest."""
    # Vector from source to dest.
    dx, dy = dest[0] - source[0], dest[1] - source[1]
    length = (dx * dx + dy * dy) ** 0.5
    if length < ARROW_MIN_LENGTH:
        # Degenerate guard — avoid div-by-zero on the unit vector below.
        return
    # Unit vector pointing toward the destination.
    ux, uy = dx / length, dy / length
    # Base center of the arrowhead: one head-length back from the tip.
    bcx, bcy = dest[0] - ux * ARROW_HEAD_LEN, dest[1] - uy * ARROW_HEAD_LEN
    # Perpendicular vector (-uy, ux) gives the two base corners.
    b1 = (bcx - uy * ARROW_HEAD_HALF_WIDTH, bcy + ux * ARROW_HEAD_HALF_WIDTH)
    b2 = (bcx + uy * ARROW_HEAD_HALF_WIDTH, bcy - ux * ARROW_HEAD_HALF_WIDTH)
    preview_surface.fill((0, 0, 0, 0))
    # The shaft is always opaque so the arrow direction is unambiguous.
    pygame.draw.line(preview_surface, color + (255,), source, dest, ARROW_LINE_WIDTH)
    # The head uses fill_alpha so previews look lighter than committed arrows.
    pygame.draw.polygon(preview_surface, color + (fill_alpha,), [dest, b1, b2], 0)
    surface.blit(preview_surface, (0, 0))


def _draw_circle(surface, center, radius, color, fill_alpha):
    # Used by circles. Same alpha recipe; int() because pygame.draw.circle
    # wants integer pixel coords.
    cx, cy, r = int(center[0]), int(center[1]), int(radius)
    preview_surface.fill((0, 0, 0, 0))
    pygame.draw.circle(preview_surface, color + (fill_alpha,), (cx, cy), r, 0)
    pygame.draw.circle(preview_surface, color + (255,), (cx, cy), r, SHAPE_OUTLINE_WIDTH)
    surface.blit(preview_surface, (0, 0))


def _diamond_vertices(center, half_w, half_h):
    # Four corners of the diamond: top, right, bottom, left.
    # Y grows downward in screen coords, so cy-half_h is the TOP.
    cx, cy = center
    return [(cx, cy - half_h), (cx + half_w, cy), (cx, cy + half_h), (cx - half_w, cy)]


def draw_shape(surface, shape, fill_alpha=SHAPE_FILL_ALPHA, highlighted=False):
    # Dispatcher: pick the right primitive based on shape type.
    # Default fill_alpha matches the live canvas; export and preview override.
    color = HIGHLIGHT_COLOR if highlighted else SHAPE_COLOR
    if shape["type"] == "square":
        # Build the pygame.Rect from center+half (Redesign-phase model).
        cx, cy = shape["center"]
        half = shape["half"]
        rect = pygame.Rect(cx - half, cy - half, half * 2, half * 2)
        _draw_filled_rect(surface, rect, color, fill_alpha, CORNER_RADIUS)
    elif shape["type"] == "diamond":
        # Diamonds need explicit vertices because they aren't axis-aligned.
        verts = _diamond_vertices(shape["center"], shape["half_w"], shape["half_h"])
        _draw_filled_polygon(surface, verts, color, fill_alpha, CORNER_RADIUS)
    elif shape["type"] == "circle":
        _draw_circle(surface, shape["center"], shape["radius"], color, fill_alpha)
    elif shape["type"] == "arrow":
        _draw_arrow(surface, shape["source"], shape["dest"], color, fill_alpha)


# Dispatch table: gesture label → builder function. Keeps the loop's preview
# branch compact (one dict lookup vs an if/elif ladder per shape).
GESTURE_BUILDERS = {
    "l_frames":       make_square,
    "triangle_frame": make_diamond,    # gesture name preserved; output is now a diamond.
    "circle_frame":   make_circle,
    "pointing":       make_arrow,
}


# ----------------------------------------------------------------------------
# Hit-testing — does the V-center of a peace_open hand sit on a shape?
# ----------------------------------------------------------------------------
#
# Each `_contains_*` takes a shape dict + a point (px, py) and returns bool.
# Math differs per shape because each one has a different "inside" definition.

def _contains_square(shape, point):
    # Axis-aligned bounding box — a point is inside iff both
    # |dx| ≤ half AND |dy| ≤ half from the center.
    cx, cy = shape["center"]
    half = shape["half"]
    return abs(point[0] - cx) <= half and abs(point[1] - cy) <= half


def _contains_diamond(shape, point):
    """Point-in-diamond: Manhattan distance from center, normalized by half-axes."""
    cx, cy = shape["center"]
    hw, hh = shape["half_w"], shape["half_h"]
    if hw == 0 or hh == 0:
        # Degenerate diamond — nothing can be inside zero area.
        return False
    # A diamond is the set of points where |dx|/hw + |dy|/hh ≤ 1.
    # That's the standard "L1 / taxicab" disk, stretched by the half-axes.
    return abs(point[0] - cx) / hw + abs(point[1] - cy) / hh <= 1.0


def _contains_circle(shape, point):
    # Standard Euclidean: inside iff squared distance ≤ squared radius.
    # We compare squared values to avoid an unnecessary sqrt.
    cx, cy = shape["center"]
    return (point[0] - cx) ** 2 + (point[1] - cy) ** 2 <= shape["radius"] ** 2


def _contains_arrow(shape, point):
    """Distance from point to source→dest segment, within tolerance band."""
    sx, sy = shape["source"]
    ex, ey = shape["dest"]
    # Segment vector and its squared length (kept squared to skip sqrt).
    seg_dx, seg_dy = ex - sx, ey - sy
    seg_len_sq = seg_dx * seg_dx + seg_dy * seg_dy
    if seg_len_sq == 0:
        # Zero-length segment — can't compute a projection.
        return False
    # Project (point - source) onto the segment, clamped to [0, 1] so the
    # closest point stays on the segment (not its infinite extension).
    t = max(0.0, min(1.0, ((point[0] - sx) * seg_dx + (point[1] - sy) * seg_dy) / seg_len_sq))
    # Closest point on the segment to `point`.
    closest_x = sx + t * seg_dx
    closest_y = sy + t * seg_dy
    # Hit if within ARROW_HIT_TOLERANCE pixels of that closest point.
    return (point[0] - closest_x) ** 2 + (point[1] - closest_y) ** 2 <= ARROW_HIT_TOLERANCE ** 2


# Dispatch table: shape type → hit-test function. Mirrors GESTURE_BUILDERS.
SHAPE_HIT_TESTS = {
    "square":   _contains_square,
    "diamond":  _contains_diamond,
    "circle":   _contains_circle,
    "arrow":    _contains_arrow,
}


def shape_contains(shape, point):
    # Public entry point. Callers don't need to know which `_contains_*` to use.
    return SHAPE_HIT_TESTS[shape["type"]](shape, point)


# ----------------------------------------------------------------------------
# Translation & bounds — for drag and out-of-bounds delete checks
# ----------------------------------------------------------------------------

def _shift(pt, dx, dy):
    # Trivial helper. Returns a new tuple so the original point stays untouched.
    return (pt[0] + dx, pt[1] + dy)


def translate_shape(shape, dx, dy):
    """Return a new shape dict with all coordinates shifted by (dx, dy)."""
    # Returns a new dict (not in-place) so debugging / logging stays sane.
    # Squares/diamonds/circles only carry a center; arrow needs both endpoints.
    t = shape["type"]
    if t == "square":
        return {"type": t, "center": _shift(shape["center"], dx, dy),
                "half": shape["half"]}
    if t == "diamond":
        return {"type": t, "center": _shift(shape["center"], dx, dy),
                "half_w": shape["half_w"], "half_h": shape["half_h"]}
    if t == "circle":
        return {"type": t, "center": _shift(shape["center"], dx, dy),
                "radius": shape["radius"]}
    if t == "arrow":
        # Both endpoints shift by the same delta — rigid translation, no rotation.
        return {"type": t, "source": _shift(shape["source"], dx, dy),
                "dest": _shift(shape["dest"], dx, dy)}


def shape_center(shape):
    """Reference point for out-of-bounds deletion check."""
    # Most shapes already store a center; arrow doesn't, so we compute one
    # from its endpoints.
    t = shape["type"]
    if t == "square":
        return shape["center"]
    if t == "diamond":
        return shape["center"]
    if t == "circle":
        return shape["center"]
    if t == "arrow":
        return ((shape["source"][0] + shape["dest"][0]) / 2,
                (shape["source"][1] + shape["dest"][1]) / 2)


def is_out_of_bounds(shape, screen_w, screen_h):
    # True if the shape's center is within DELETE_THRESHOLD px of any edge.
    # Used by the drag loop to show the warning border and decide on release.
    cx, cy = shape_center(shape)
    return (cx < DELETE_THRESHOLD or cx > screen_w - DELETE_THRESHOLD
            or cy < DELETE_THRESHOLD or cy > screen_h - DELETE_THRESHOLD)
