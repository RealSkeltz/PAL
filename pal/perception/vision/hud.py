"""Draws tracked objects onto a frame.

Corner brackets rather than a full outline: they read as a reticle, and with no
continuous edge any residual wobble is far less obvious. Each track composites
over its own region so it can fade out independently of the others.

This is the interim renderer. Everything here is destined to move to an SVG
overlay in the browser, where the graphics stop being JPEG-compressed and box
motion can be interpolated by CSS instead of by hand.
"""

from collections import namedtuple

import cv2

HUD_COLOR = (192, 211, 125)     # the preview's accent teal #7dd3c0, in BGR
SHADOW_COLOR = (20, 20, 20)
CROSSHAIR_COLOR = (150, 150, 150)
HUD_ALPHA = 0.85
ARM_FRACTION = 0.18             # corner arm length, as a share of the short edge
MIN_ARM = 10
PAD = 3                         # room for stroke and text shadow when compositing
CAPTION_GAP = 6                 # breathing room between box and caption
EDGE_MARGIN = 6                 # keeps a nudged caption off the frame border
# Only prominent objects get labelled. A caption on every distant figure turns a
# crowd into soup, and at a legible type size the text would dwarf the box it
# names. Prominence is the gate; the fit ratio just stops extreme overhang.
MIN_CAPTION_SPAN = 0.25         # box height as a share of frame height
CAPTION_FIT = 2.0               # caption width as a multiple of box width
FONT = cv2.FONT_HERSHEY_SIMPLEX

# Caption size tracks capture resolution — a scale tuned against a 760px-tall
# frame is illegible at 1080p and overbearing on a thumbnail.
REFERENCE_HEIGHT = 760
SCALE_RANGE = (0.7, 1.3)

Style = namedtuple("Style", "font_scale font_weight line_weight")


def annotate(frame, tracks):
    """Return a copy of `frame` with a reticle drawn for each visible track."""
    img = frame.copy()
    h, w = img.shape[:2]
    style = _style_for(h)

    if tracks:
        overlay = img.copy()
        laid_out = [_layout(track, img.shape, style) for track in tracks]
        for reticle, caption, _ in laid_out:
            _draw_track(overlay, reticle, caption, img.shape, style)
        for track, (_, _, bounds) in zip(tracks, laid_out):
            _blend_region(img, overlay, bounds, track.opacity * HUD_ALPHA)

    cv2.drawMarker(img, (w // 2, h // 2), CROSSHAIR_COLOR, cv2.MARKER_CROSS, 14, 1, cv2.LINE_AA)
    return img


# ----- internals -----


def _style_for(frame_height):
    low, high = SCALE_RANGE
    scale = min(high, max(low, frame_height / REFERENCE_HEIGHT))
    return Style(scale, max(2, round(scale * 2)), max(2, round(scale * 2)))


def _bracket_corners(x1, y1, x2, y2, shape):
    """Yield the segments for each corner that actually sits inside the frame.

    An object running past the edge of view has corners beyond it. Drawing those
    anyway leaves the on-screen half of each arm pinned to the frame border as a
    stub; dropping the corner instead leaves the reticle open on that side, which
    reads correctly as "this continues past the edge".
    """
    h, w = shape[:2]
    arm = max(MIN_ARM, int(min(x2 - x1, y2 - y1) * ARM_FRACTION))
    for x, dx in ((x1, arm), (x2, -arm)):
        if not 0 <= x < w:
            continue
        for y, dy in ((y1, arm), (y2, -arm)):
            if not 0 <= y < h:
                continue
            yield (x, y), (x + dx, y)
            yield (x, y), (x, y + dy)


def _caption(track, box, shape, style):
    """The caption for a track, or None if the object is not prominent enough.

    Hershey fonts are ASCII-only, so no interpunct until the SVG overlay lands.
    """
    x1, y1, x2, y2 = box
    if (y2 - y1) < shape[0] * MIN_CAPTION_SPAN:
        return None

    text = f"{track.label} {track.conf:.2f}"
    (width, height), baseline = cv2.getTextSize(text, FONT, style.font_scale, style.font_weight)
    if width > (x2 - x1) * CAPTION_FIT:
        return None
    return text, width, height, baseline


def _layout(track, shape, style):
    """Reticle rect, caption, and the union of the two clipped to the frame."""
    h, w = shape[:2]
    x1, y1, x2, y2 = (int(v) for v in track.xyxy)

    caption = _caption(track, (x1, y1, x2, y2), shape, style)
    left, top, right, bottom = x1, y1, x2, y2
    text, origin = None, None

    if caption is not None:
        text, text_w, text_h, baseline = caption
        # Sit the caption under the box, but keep it on screen for objects that
        # run past the bottom edge or off to the left.
        rightmost = max(EDGE_MARGIN, w - text_w - EDGE_MARGIN)
        origin = (min(max(EDGE_MARGIN, x1), rightmost),
                  min(y2 + text_h + CAPTION_GAP, h - baseline - EDGE_MARGIN))
        left = min(left, origin[0])
        top = min(top, origin[1] - text_h)
        right = max(right, origin[0] + text_w)
        bottom = max(bottom, origin[1] + baseline)

    bounds = (max(0, left - PAD), max(0, top - PAD),
              min(w, right + PAD), min(h, bottom + PAD))
    return (x1, y1, x2, y2), (text, origin), bounds


def _draw_track(overlay, reticle, caption, shape, style):
    for start, end in _bracket_corners(*reticle, shape):
        cv2.line(overlay, start, end, HUD_COLOR, style.line_weight, cv2.LINE_AA)

    text, origin = caption
    if text is None:
        return
    # Shadow first: thin accent-coloured type over live video is hard to read
    # against a light or busy background.
    cv2.putText(overlay, text, (origin[0] + 1, origin[1] + 1), FONT, style.font_scale,
                SHADOW_COLOR, style.font_weight + 1, cv2.LINE_AA)
    cv2.putText(overlay, text, origin, FONT, style.font_scale, HUD_COLOR,
                style.font_weight, cv2.LINE_AA)


def _blend_region(img, overlay, bounds, alpha):
    """Composite just the area a track occupies, so each can fade on its own."""
    x1, y1, x2, y2 = bounds
    if x2 <= x1 or y2 <= y1:
        return
    region = img[y1:y2, x1:x2]
    cv2.addWeighted(overlay[y1:y2, x1:x2], alpha, region, 1 - alpha, 0, region)
