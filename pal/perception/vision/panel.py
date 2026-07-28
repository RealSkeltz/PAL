"""The corner status panel and the tool badge.

Chrome rather than annotation: what the agent is doing and how long the last few
steps took, gathered into one plate so it reads as an instrument panel instead of
text scattered over the frame.

Text is drawn at full opacity onto a translucent plate rather than being blended
along with it — alpha'd type over live video is the difference between legible
and not.
"""

import math

import cv2

from pal.status import TOOL_BADGE_SECONDS, State

FONT = cv2.FONT_HERSHEY_SIMPLEX

MARGIN = 16                 # gap between plate and frame edge
PAD = 10                    # plate interior padding
ROW_GAP = 7
COLUMN_GAP = 26             # minimum space between a row's label and its value
CORNER_RADIUS = 8
PLATE_COLOR = (18, 18, 20)
PLATE_ALPHA = 0.72
LABEL_COLOR = (145, 148, 152)
VALUE_COLOR = (230, 231, 234)
DOT_RADIUS = 4
DOT_GAP = 9

# Matches the accents the preview page already uses for the same concepts, so the
# frame and the event log agree on what each colour means.
STATE_COLORS = {
    State.IDLE: (140, 140, 140),
    State.LISTENING: (192, 211, 125),    # #7dd3c0 — as "Heard"
    State.THINKING: (212, 148, 183),     # #b794d4
    State.SPEAKING: (110, 169, 201),     # #c9a96e — as "Said"
}

TOOL_COLOR = (212, 148, 183)
TOOL_RADIUS = 15
TOOL_TEETH = 8
TOOL_FADE_AT = 0.7          # share of the badge's life before it starts fading


def draw(img, snapshot, style):
    """Overlay the status plate and, if a tool just ran, its badge."""
    _draw_plate(img, snapshot, style)
    if snapshot.tool:
        _draw_tool_badge(img, snapshot, style)


# ----- status plate -----


def _draw_plate(img, snap, style):
    h, w = img.shape[:2]
    head_scale = style.font_scale * 0.9
    row_scale = style.font_scale * 0.78
    row_weight = max(1, style.font_weight - 1)

    head = snap.state.value.upper()
    (head_w, head_h), _ = cv2.getTextSize(head, FONT, head_scale, style.font_weight)

    rows = []
    for label, milliseconds in snap.timings:
        value = _format_ms(milliseconds)
        (label_w, label_h), _ = cv2.getTextSize(label, FONT, row_scale, row_weight)
        (value_w, value_h), _ = cv2.getTextSize(value, FONT, row_scale, row_weight)
        rows.append((label, value, label_w, value_w, max(label_h, value_h)))

    inner_w = max([DOT_RADIUS * 2 + DOT_GAP + head_w]
                  + [lw + COLUMN_GAP + vw for _, _, lw, vw, _ in rows])
    inner_h = head_h + sum(row_h + ROW_GAP for *_, row_h in rows)

    x1, y1 = MARGIN, MARGIN
    x2, y2 = min(w, x1 + inner_w + PAD * 2), min(h, y1 + inner_h + PAD * 2)
    if x2 <= x1 or y2 <= y1:
        return
    _plate(img, (x1, y1, x2, y2))

    # Header: state dot, then the state itself.
    baseline = y1 + PAD + head_h
    cv2.circle(img, (x1 + PAD + DOT_RADIUS, baseline - head_h // 2), DOT_RADIUS,
               STATE_COLORS.get(snap.state, LABEL_COLOR), -1, cv2.LINE_AA)
    cv2.putText(img, head, (x1 + PAD + DOT_RADIUS * 2 + DOT_GAP, baseline), FONT,
                head_scale, STATE_COLORS.get(snap.state, LABEL_COLOR),
                style.font_weight, cv2.LINE_AA)

    # Rows: label flush left, value flush right.
    right = x2 - PAD
    for label, value, _, value_w, row_h in rows:
        baseline += row_h + ROW_GAP
        cv2.putText(img, label, (x1 + PAD, baseline), FONT, row_scale,
                    LABEL_COLOR, row_weight, cv2.LINE_AA)
        cv2.putText(img, value, (right - value_w, baseline), FONT, row_scale,
                    VALUE_COLOR, row_weight, cv2.LINE_AA)


def _plate(img, rect, alpha=PLATE_ALPHA):
    """Blend a translucent rounded plate into the frame, text-free."""
    x1, y1, x2, y2 = rect
    roi = img[y1:y2, x1:x2]
    filled = roi.copy()
    _rounded_rect(filled, 0, 0, x2 - x1 - 1, y2 - y1 - 1, PLATE_COLOR, CORNER_RADIUS)
    cv2.addWeighted(filled, alpha, roi, 1 - alpha, 0, roi)


def _rounded_rect(img, x1, y1, x2, y2, color, radius):
    cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, -1)
    cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, -1)
    for cx, cy, start in ((x1 + radius, y1 + radius, 180), (x2 - radius, y1 + radius, 270),
                          (x2 - radius, y2 - radius, 0), (x1 + radius, y2 - radius, 90)):
        cv2.ellipse(img, (cx, cy), (radius, radius), start, 0, 90, color, -1, cv2.LINE_AA)


def _format_ms(milliseconds):
    if milliseconds >= 1000:
        return f"{milliseconds / 1000:.1f}s"
    return f"{milliseconds:.0f}ms"


# ----- tool badge -----


def _draw_tool_badge(img, snap, style):
    """A gear in the opposite corner from the status plate, fading as it ages.

    Plated like the status panel: an unbacked gear vanishes against a bright
    background, and the badge is worth nothing if it cannot be seen.
    """
    h, w = img.shape[:2]
    scale = style.font_scale * 0.72
    weight = max(1, style.font_weight - 1)
    fade = _badge_alpha(snap.tool_age)
    if fade <= 0:
        return

    (text_w, text_h), _ = cv2.getTextSize(snap.tool, FONT, scale, weight)
    inner_w = max(text_w, (TOOL_RADIUS + 4) * 2)
    x2, y1 = w - MARGIN, MARGIN
    x1 = max(0, x2 - inner_w - PAD * 2)
    y2 = min(h, y1 + (TOOL_RADIUS + 4) * 2 + ROW_GAP + text_h + PAD * 2)
    if x2 <= x1 or y2 <= y1:
        return

    # Plate first, then the graphics on top — both scaled by the same fade so the
    # badge dissolves as a single object.
    _plate(img, (x1, y1, x2, y2), alpha=PLATE_ALPHA * fade)

    roi = img[y1:y2, x1:x2]
    overlay = roi.copy()
    span = x2 - x1
    _gear(overlay, (span // 2, PAD + TOOL_RADIUS + 4), TOOL_RADIUS, TOOL_COLOR,
          max(1, style.line_weight - 1))
    cv2.putText(overlay, snap.tool, ((span - text_w) // 2, y2 - y1 - PAD), FONT,
                scale, TOOL_COLOR, weight, cv2.LINE_AA)
    cv2.addWeighted(overlay, fade, roi, 1 - fade, 0, roi)


def _badge_alpha(age):
    fade_start = TOOL_BADGE_SECONDS * TOOL_FADE_AT
    if age <= fade_start:
        return 1.0
    remaining = (TOOL_BADGE_SECONDS - age) / (TOOL_BADGE_SECONDS - fade_start)
    return max(0.0, min(1.0, remaining))


def _gear(img, centre, radius, color, weight):
    cx, cy = centre
    cv2.circle(img, centre, radius, color, weight, cv2.LINE_AA)
    cv2.circle(img, centre, max(2, radius // 3), color, weight, cv2.LINE_AA)
    for i in range(TOOL_TEETH):
        angle = 2 * math.pi * i / TOOL_TEETH
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        cv2.line(img,
                 (int(cx + cos_a * radius), int(cy + sin_a * radius)),
                 (int(cx + cos_a * (radius + 4)), int(cy + sin_a * (radius + 4))),
                 color, weight, cv2.LINE_AA)
