"""Draws tracked objects onto a frame.

Corner brackets rather than a full outline: they read as a reticle, and with no
continuous edge any residual wobble is far less obvious. Each track composites
over its own region so it can fade out independently of the others.

Geometry is OpenCV, which anti-aliases lines; type is Pillow, which has real
letterforms. All text for the frame — reticle captions and the status plate
alike — is collected on one layer and blended in a single pass at the end.
"""

import cv2

from pal.hud import panel
from pal.hud import style as S
from pal.hud.typography import TextLayer, cap_height, font, measure

RETICLE_COLOR = S.bgr(S.ACCENT)
CROSSHAIR_COLOR = S.bgr(S.TEXT_FAINT)
SHADOW = (0, 0, 0)

HUD_ALPHA = 0.9
ARM_FRACTION = 0.18             # corner arm length, as a share of the short edge
MIN_ARM = 10
PAD = 3                         # room for the anti-aliased stroke when compositing
CAPTION_GAP = S.UNIT * 2        # box to caption
CONF_GAP = S.UNIT * 2           # label to confidence

# Only prominent objects get labelled. A caption on every distant figure turns a
# crowd into soup, and at a legible type size the text would dwarf the box it
# names. Prominence is the gate; the fit ratio just stops extreme overhang.
MIN_CAPTION_SPAN = 0.25         # box height as a share of frame height
CAPTION_FIT = 2.0               # caption width as a multiple of box width

CROSSHAIR_ARM = 6
CROSSHAIR_GAP = 3


def annotate(frame, tracks, snapshot=None):
    """Return a copy of `frame` with a reticle per visible track, plus chrome."""
    img = frame.copy()
    height, width = img.shape[:2]
    layer = TextLayer(img.shape)
    line_weight = max(2, round(2 * height / S.REFERENCE_HEIGHT))

    if tracks:
        overlay = img.copy()
        for track in tracks:
            _draw_reticle(overlay, track, img.shape, line_weight)
        for track in tracks:
            _blend_region(img, overlay, track, track.opacity * HUD_ALPHA)
        for track in tracks:
            _draw_caption(layer, track, img.shape, height)

    _crosshair(img, width // 2, height // 2)

    if snapshot is not None:
        panel.draw(img, layer, snapshot)

    layer.composite(img)
    return img


# ----- reticles -----


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


def _draw_reticle(overlay, track, shape, line_weight):
    box = tuple(int(v) for v in track.xyxy)
    for start, end in _bracket_corners(*box, shape):
        cv2.line(overlay, start, end, RETICLE_COLOR, line_weight, cv2.LINE_AA)


def _blend_region(img, overlay, track, alpha):
    """Composite just the area a track occupies, so each can fade on its own."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = (int(v) for v in track.xyxy)
    x1, y1 = max(0, x1 - PAD), max(0, y1 - PAD)
    x2, y2 = min(w, x2 + PAD), min(h, y2 + PAD)
    if x2 <= x1 or y2 <= y1:
        return
    region = img[y1:y2, x1:x2]
    cv2.addWeighted(overlay[y1:y2, x1:x2], alpha, region, 1 - alpha, 0, region)


# ----- captions -----


def _draw_caption(layer, track, shape, height):
    """Label in tracked caps, confidence in mono — the panel's typographic rules.

    Text goes on the shared layer rather than into the frame, so it is never
    clipped by the track's own composite region.
    """
    frame_h, frame_w = shape[:2]
    x1, y1, x2, y2 = (int(v) for v in track.xyxy)
    if (y2 - y1) < frame_h * MIN_CAPTION_SPAN:
        return

    size = S.scaled(S.SIZE_CAPTION, height)
    label_face = font("sans", size, "Semibold")
    conf_face = font("mono", size, "Medium")
    tracking = S.tracking_for(size)

    label = track.label.upper()
    conf = f"{track.conf:.2f}"
    label_w = measure(label, label_face, tracking)
    conf_w = measure(conf, conf_face)
    total_w = label_w + CONF_GAP + conf_w
    if total_w > (x2 - x1) * CAPTION_FIT:
        return

    # Keep the caption on screen for objects running past an edge.
    x = min(max(S.MARGIN, x1), max(S.MARGIN, frame_w - total_w - S.MARGIN))
    y = min(y2 + CAPTION_GAP + cap_height(label_face), frame_h - S.UNIT)
    alpha = round(255 * track.opacity * HUD_ALPHA)

    # Haloed: unlike the status plate, a caption has no backing, so the type has
    # to hold its own against whatever the camera is pointed at.
    layer.text((x, y), label, label_face, S.ACCENT, tracking, alpha=alpha, halo=SHADOW)
    layer.text((x + CONF_GAP + label_w, y), conf, conf_face, S.TEXT,
               alpha=alpha, halo=SHADOW)


def _crosshair(img, cx, cy):
    """Four ticks around an open centre — a bare plus sign reads as a placeholder."""
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        start = (cx + dx * CROSSHAIR_GAP, cy + dy * CROSSHAIR_GAP)
        end = (cx + dx * (CROSSHAIR_GAP + CROSSHAIR_ARM),
               cy + dy * (CROSSHAIR_GAP + CROSSHAIR_ARM))
        cv2.line(img, start, end, CROSSHAIR_COLOR, 1, cv2.LINE_AA)
