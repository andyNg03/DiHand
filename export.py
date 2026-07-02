"""
export.py — save the diagram to disk.

Two files per export, sharing a timestamp:
  dihand_YYYYMMDD_HHMMSS.png         — rasterized snapshot for sharing.
  dihand_YYYYMMDD_HHMMSS.excalidraw  — JSON, openable at excalidraw.com for further editing.

The driver calls export_diagram(shapes, w, h) once when the user holds the
export gesture for long enough. This module knows nothing about the trigger
(hold timer, progress arc, etc.) — those live in app.py with the loop.
"""

import json
import random
import string
import time

import pygame

# We import `shape` as a module (not just functions) because the PNG render
# temporarily rebinds shape.preview_surface — see export_diagram below.
import shape
from shape import draw_shape


def export_diagram(committed_shapes, screen_w, screen_h):
    """Save committed shapes as PNG on a white background, fully opaque."""
    # Trick: draw_shape always paints onto shape.preview_surface, which is
    # sized to the screen. We temporarily swap in a fresh surface so the
    # export draws are clean (no leftover pixels from the live canvas).
    # We rebind the attribute on the shape module (no `global` needed —
    # the surface lives in shape.py, not here).
    saved_ps = shape.preview_surface
    shape.preview_surface = pygame.Surface((screen_w, screen_h), pygame.SRCALPHA)
    # Opaque white background — the export is the "real" diagram, not a webcam overlay.
    surface = pygame.Surface((screen_w, screen_h))
    surface.fill((255, 255, 255))
    # Loop var is `s` (not `shape`) so it doesn't shadow the `shape` module above.
    for s in committed_shapes:
        # fill_alpha=255 overrides the live-canvas translucency.
        draw_shape(surface, s, fill_alpha=255)
    # Restore the original preview_surface for the next frame's drawing.
    shape.preview_surface = saved_ps
    # Two files with the same timestamp so they're easy to find as a pair.
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    png_name = f"dihand_{timestamp}.png"
    pygame.image.save(surface, png_name)
    excalidraw_name = f"dihand_{timestamp}.excalidraw"
    _export_excalidraw(committed_shapes, excalidraw_name)
    return png_name


def _random_id():
    # Excalidraw expects each element to have a unique 20-char alphanumeric ID.
    return ''.join(random.choices(string.ascii_letters + string.digits, k=20))


def _excalidraw_base(etype, x, y, w, h):
    """Shared fields for every Excalidraw element."""
    # Excalidraw elements are heavy dicts with many required fields. This
    # builder fills in everything common; per-shape code adds type-specific
    # extras (e.g. arrow points) on top. All visual fields match the live canvas.
    return {
        "id": _random_id(),                # unique per element.
        "type": etype,                     # "rectangle" | "diamond" | "ellipse" | "arrow".
        "x": x,                            # top-left x of the element's bounding box.
        "y": y,                            # top-left y.
        "width": w,
        "height": h,
        "angle": 0,                        # no rotation; matches our axis-aligned model.
        "strokeColor": "#d92243",          # outline color = DiHand's SHAPE_COLOR.
        "backgroundColor": "#d92243",      # fill = same color (Excalidraw "solid" needs a bg).
        "fillStyle": "solid",              # no hatch / cross-hatch.
        "strokeWidth": 2,                  # matches SHAPE_OUTLINE_WIDTH on the live canvas.
        "strokeStyle": "solid",            # no dashed/dotted lines.
        "roughness": 0,                    # clean, not Excalidraw's default sketchy style.
        "opacity": 100,                    # fully opaque — exports aren't translucent.
        "groupIds": [],                    # not grouped.
        "roundness": None,                 # default corner rounding (Excalidraw decides).
        "seed": random.randint(1, 2_000_000_000),         # determines sketch randomness; unused at roughness=0.
        "version": 1,
        "versionNonce": random.randint(1, 2_000_000_000), # Excalidraw uses these for collab merge tracking.
        "isDeleted": False,
        "boundElements": None,             # not bound to text or another element.
        "updated": int(time.time() * 1000),  # last-modified ms timestamp.
        "link": None,
        "locked": False,
    }


def _shape_to_excalidraw(shape):
    """Convert one DiHand shape dict into an Excalidraw element dict."""
    # Note: `shape` here is the parameter name (a shape dict). It shadows the
    # `shape` module within this function only — fine because we don't touch
    # the module here.
    t = shape["type"]

    if t == "square":
        # Map to Excalidraw "rectangle". We supply top-left (x,y) and size,
        # so subtract `half` from the center to get the corner.
        cx, cy = shape["center"]
        half = shape["half"]
        return _excalidraw_base("rectangle", cx - half, cy - half, half * 2, half * 2)

    if t == "diamond":
        # Excalidraw has a native "diamond" type — same bounding-box convention.
        cx, cy = shape["center"]
        hw, hh = shape["half_w"], shape["half_h"]
        return _excalidraw_base("diamond", cx - hw, cy - hh, hw * 2, hh * 2)

    if t == "circle":
        # Excalidraw uses "ellipse" for circles too. Bounding box of an
        # ellipse with width=2r, height=2r = a circle.
        cx, cy = shape["center"]
        r = shape["radius"]
        return _excalidraw_base("ellipse", cx - r, cy - r, r * 2, r * 2)

    if t == "arrow":
        # Arrows are special: (x,y) is the source, "points" is a list of
        # offsets from that origin. [0,0] = source, [dx,dy] = dest.
        sx, sy = shape["source"]
        dx, dy = shape["dest"]
        el = _excalidraw_base("arrow", sx, sy, dx - sx, dy - sy)
        el["points"] = [[0, 0], [dx - sx, dy - sy]]
        el["startArrowhead"] = None    # no head at source.
        el["endArrowhead"] = "arrow"   # filled triangle at dest — matches our render.
        el["startBinding"] = None      # not bound to any shape (no auto-connect).
        el["endBinding"] = None
        return el


def _export_excalidraw(committed_shapes, filename):
    """Write an .excalidraw JSON file that excalidraw.com can open directly."""
    # Convert every shape into an Excalidraw element dict.
    elements = [_shape_to_excalidraw(s) for s in committed_shapes]
    # Top-level Excalidraw document schema. `version: 2` is the current
    # format excalidraw.com expects when importing.
    doc = {
        "type": "excalidraw",
        "version": 2,
        "source": "DiHand",                # identifies where the file came from.
        "elements": elements,
        "appState": {"viewBackgroundColor": "#ffffff"},   # white canvas, matches our PNG.
        "files": {},                                      # no embedded images.
    }
    with open(filename, "w") as f:
        json.dump(doc, f, indent=2)
