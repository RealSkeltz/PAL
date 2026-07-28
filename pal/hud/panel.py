"""The corner status plate and the tool badge.

Chrome rather than annotation: what the agent is doing and how long the last few
steps took, gathered onto one plate so it reads as an instrument rather than text
scattered over the frame.

Geometry is OpenCV (anti-aliased), type is Pillow (real letterforms). Values are
split into figure and unit so the number carries the weight and the unit recedes,
and figures are set in a mono face so the readout does not shift as they change.
"""

import math

import cv2

from pal.hud import style as S
from pal.hud.typography import cap_height, font, measure
from pal.status import TOOL_BADGE_SECONDS, State

STATE_COLORS = {
    State.IDLE: S.STATE_IDLE,
    State.LISTENING: S.STATE_LISTENING,
    State.THINKING: S.STATE_THINKING,
    State.SPEAKING: S.STATE_SPEAKING,
}

CORNER_RADIUS = 3
BRACKET_LEN = 10                # arm length of the plate's corner marks
DOT_RADIUS = 3
DOT_GAP = S.UNIT * 3
GLOW_STEPS = ((2.8, 0.10), (1.9, 0.22))   # (radius multiple, alpha)
TOOL_RADIUS = 11
TOOL_TICKS = 4
TOOL_FADE_AT = 0.7


def draw(img, layer, snapshot):
    """Overlay the status plate and, if a tool just ran, its badge.

    Text is added to the caller's layer rather than composited here, so the whole
    frame's type lands in a single blend.
    """
    height = img.shape[0]
    _draw_plate(img, layer, snapshot, height)
    fade = _badge_alpha(snapshot.tool_age) if snapshot.tool else 0.0
    if fade > 0:
        _draw_tool_badge(img, layer, snapshot, height, fade)


# ----- status plate -----


def _draw_plate(img, layer, snap, height):
    frame_h, frame_w = img.shape[:2]

    header_face = font("sans", S.scaled(S.SIZE_HEADER, height), "Semibold")
    label_face = font("sans", S.scaled(S.SIZE_LABEL, height), "Medium")
    value_face = font("mono", S.scaled(S.SIZE_VALUE, height), "Medium")
    unit_face = font("mono", S.scaled(S.SIZE_UNIT, height), "Medium")

    header = snap.state.value.upper()
    header_tracking = S.tracking_for(header_face.size)
    header_w = measure(header, header_face, header_tracking)
    header_h = cap_height(header_face)

    label_tracking = S.tracking_for(label_face.size)
    rows = []
    for label, milliseconds in snap.timings:
        figure, unit = _format_ms(milliseconds)
        label = label.replace("_", " ").upper()
        rows.append({
            "label": label,
            "label_w": measure(label, label_face, label_tracking),
            "figure": figure,
            "figure_w": measure(figure, value_face),
            "unit": unit,
            "unit_w": measure(unit, unit_face),
        })

    row_h = cap_height(value_face)
    inner_w = max(
        [DOT_RADIUS * 2 + DOT_GAP + header_w]
        + [r["label_w"] + S.COLUMN_GAP + r["figure_w"] + S.UNIT + r["unit_w"] for r in rows]
    )
    inner_h = header_h
    if rows:
        inner_h += S.RULE_GAP * 2 + 1 + len(rows) * row_h + (len(rows) - 1) * S.ROW_GAP

    x1, y1 = S.MARGIN, S.MARGIN
    x2 = min(frame_w, x1 + inner_w + S.PAD_X * 2)
    y2 = min(frame_h, y1 + inner_h + S.PAD_Y * 2)
    if x2 <= x1 or y2 <= y1:
        return

    _plate(img, (x1, y1, x2, y2))
    _corner_brackets(img, (x1, y1, x2, y2), S.bgr(S.ACCENT), S.BRACKET_ALPHA)

    accent = STATE_COLORS.get(snap.state, S.TEXT_DIM)
    baseline = y1 + S.PAD_Y + header_h
    dot_centre = (x1 + S.PAD_X + DOT_RADIUS, baseline - header_h // 2)
    _glow_dot(img, dot_centre, DOT_RADIUS, S.bgr(accent))
    layer.text((x1 + S.PAD_X + DOT_RADIUS * 2 + DOT_GAP, baseline), header,
               header_face, accent, header_tracking)

    if not rows:
        return

    rule_y = baseline + S.RULE_GAP
    _rule(img, x1 + S.PAD_X, x2 - S.PAD_X, rule_y)

    right = x2 - S.PAD_X
    baseline = rule_y + S.RULE_GAP + row_h
    for row in rows:
        layer.text((x1 + S.PAD_X, baseline), row["label"], label_face,
                   S.TEXT_DIM, label_tracking)
        unit_x = right - row["unit_w"]
        layer.text((unit_x, baseline), row["unit"], unit_face, S.TEXT_FAINT)
        layer.text((unit_x - S.UNIT - row["figure_w"], baseline), row["figure"],
                   value_face, S.TEXT)
        baseline += row_h + S.ROW_GAP


def _plate(img, rect, alpha=S.PLATE_ALPHA):
    x1, y1, x2, y2 = rect
    roi = img[y1:y2, x1:x2]
    filled = roi.copy()
    _rounded_rect(filled, 0, 0, x2 - x1 - 1, y2 - y1 - 1, S.bgr(S.PLATE), CORNER_RADIUS)
    cv2.addWeighted(filled, alpha, roi, 1 - alpha, 0, roi)


def _rounded_rect(img, x1, y1, x2, y2, color, radius):
    cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, -1)
    cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, -1)
    for cx, cy, start in ((x1 + radius, y1 + radius, 180), (x2 - radius, y1 + radius, 270),
                          (x2 - radius, y2 - radius, 0), (x1 + radius, y2 - radius, 90)):
        cv2.ellipse(img, (cx, cy), (radius, radius), start, 0, 90, color, -1, cv2.LINE_AA)


def _corner_brackets(img, rect, color, alpha):
    """Accent marks at the plate's corners, echoing the reticles on the subjects."""
    x1, y1, x2, y2 = rect
    overlay = img.copy()
    for x, dx in ((x1, BRACKET_LEN), (x2 - 1, -BRACKET_LEN)):
        for y, dy in ((y1, BRACKET_LEN), (y2 - 1, -BRACKET_LEN)):
            cv2.line(overlay, (x, y), (x + dx, y), color, 1, cv2.LINE_AA)
            cv2.line(overlay, (x, y), (x, y + dy), color, 1, cv2.LINE_AA)
    roi, source = img[y1:y2, x1:x2], overlay[y1:y2, x1:x2]
    cv2.addWeighted(source, alpha, roi, 1 - alpha, 0, roi)


def _rule(img, x1, x2, y):
    overlay = img.copy()
    cv2.line(overlay, (x1, y), (x2, y), S.bgr(S.RULE), 1, cv2.LINE_AA)
    roi, source = img[y:y + 1, x1:x2], overlay[y:y + 1, x1:x2]
    cv2.addWeighted(source, S.RULE_ALPHA, roi, 1 - S.RULE_ALPHA, 0, roi)


def _glow_dot(img, centre, radius, color):
    """A lit dot: concentric translucent rings rather than a flat disc.

    Scoped to the dot's own neighbourhood — the glow covers a few dozen pixels,
    so copying the whole frame per ring is pure waste.
    """
    cx, cy = centre
    reach = int(radius * max(m for m, _ in GLOW_STEPS)) + 2
    x1, y1 = max(0, cx - reach), max(0, cy - reach)
    x2, y2 = min(img.shape[1], cx + reach), min(img.shape[0], cy + reach)
    if x2 <= x1 or y2 <= y1:
        return

    roi = img[y1:y2, x1:x2]
    local = (cx - x1, cy - y1)
    for multiple, alpha in GLOW_STEPS:
        overlay = roi.copy()
        cv2.circle(overlay, local, int(radius * multiple), color, -1, cv2.LINE_AA)
        cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0, roi)
    cv2.circle(roi, local, radius, color, -1, cv2.LINE_AA)


def _format_ms(milliseconds):
    """Split into figure and unit so they can be weighted differently."""
    if milliseconds >= 1000:
        return f"{milliseconds / 1000:.2f}", "s"
    return f"{milliseconds:.0f}", "ms"


# ----- tool badge -----


def _draw_tool_badge(img, layer, snap, height, fade):
    """A mark in the opposite corner from the plate, fading as it ages."""
    frame_h, frame_w = img.shape[:2]
    label_face = font("sans", S.scaled(S.SIZE_LABEL, height), "Medium")
    tracking = S.tracking_for(label_face.size)

    name = snap.tool.replace("_", " ").upper()
    name_w = measure(name, label_face, tracking)
    inner_w = max(name_w, TOOL_RADIUS * 2 + S.UNIT * 2)
    inner_h = TOOL_RADIUS * 2 + S.ROW_GAP + cap_height(label_face)

    x2, y1 = frame_w - S.MARGIN, S.MARGIN
    x1 = max(0, x2 - inner_w - S.PAD_X * 2)
    y2 = min(frame_h, y1 + inner_h + S.PAD_Y * 2)
    if x2 <= x1 or y2 <= y1:
        return

    _plate(img, (x1, y1, x2, y2), alpha=S.PLATE_ALPHA * fade)
    _corner_brackets(img, (x1, y1, x2, y2), S.bgr(S.TOOL), S.BRACKET_ALPHA * fade)

    centre = ((x1 + x2) // 2, y1 + S.PAD_Y + TOOL_RADIUS)
    _tool_mark(img, centre, TOOL_RADIUS, S.bgr(S.TOOL), fade)
    layer.text(((x1 + x2 - name_w) // 2, y2 - S.PAD_Y), name, label_face,
               S.TOOL, tracking, alpha=round(255 * fade))


def _tool_mark(img, centre, radius, color, fade):
    """A ring with four ticks — a gear reads as clip art at this size."""
    overlay = img.copy()
    cx, cy = centre
    cv2.circle(overlay, centre, radius, color, 1, cv2.LINE_AA)
    cv2.circle(overlay, centre, max(2, radius // 3), color, -1, cv2.LINE_AA)
    for i in range(TOOL_TICKS):
        angle = 2 * math.pi * i / TOOL_TICKS + math.pi / 4
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        cv2.line(overlay,
                 (int(cx + cos_a * (radius + 2)), int(cy + sin_a * (radius + 2))),
                 (int(cx + cos_a * (radius + 5)), int(cy + sin_a * (radius + 5))),
                 color, 1, cv2.LINE_AA)
    cv2.addWeighted(overlay, fade, img, 1 - fade, 0, img)


def _badge_alpha(age):
    fade_start = TOOL_BADGE_SECONDS * TOOL_FADE_AT
    if age <= fade_start:
        return 1.0
    remaining = (TOOL_BADGE_SECONDS - age) / (TOOL_BADGE_SECONDS - fade_start)
    return max(0.0, min(1.0, remaining))
