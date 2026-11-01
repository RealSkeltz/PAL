"""Drawing recognized objects on a frame: a see-through mask, an outline, and a label."""
import cv2
import numpy as np

# BGR, the preview page's colours
IDENTIFYING = (150, 150, 150)
STORED = (112, 158, 25)       # green
RECOGNIZED = (229, 135, 57)   # blue
PALETTE = [(0, 133, 201), (129, 81, 213), (233, 133, 144), (38, 89, 217), (180, 160, 40)]


def render(r, styles: list[tuple]) -> np.ndarray:
    """The frame of result `r` with each object's mask, outline and label. `styles` has one
    (colour, label, progress or None, flashing) per box, in box order."""
    img = r.orig_img.copy()
    if not len(r.boxes):
        return img
    shapes = [(box, outline(r, i), *style) for i, (box, style) in enumerate(zip(r.boxes, styles))]

    fill = img.copy()
    for _, contours, color, *_ in shapes:
        cv2.drawContours(fill, contours, -1, color, -1)
    img = cv2.addWeighted(fill, 0.3, img, 0.7, 0)

    for box, contours, color, text, progress, flashing in shapes:
        cv2.drawContours(img, contours, -1, color, 4 if flashing else 2, cv2.LINE_AA)
        if contours:  # the label sits on the object's highest point
            top = min((tuple(p[0]) for c in contours for p in c), key=lambda p: p[1])
            pill(img, text, top[0] - 40, top[1], color, progress)
        else:
            x1, y1 = box.xyxy[0][:2].tolist()
            pill(img, text, x1, y1, color, progress)
    return img


def outline(r, i) -> list[np.ndarray]:
    """The contours of object `i`'s mask, in frame coordinates; empty without a mask."""
    if r.masks is None:
        return []
    h, w = r.orig_img.shape[:2]
    mask = cv2.resize(r.masks.data[i].cpu().numpy(), (w, h)) > 0.5
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c for c in contours if cv2.contourArea(c) > 0.002 * h * w]  # drop specks


def pill(img, text, x, y, color, progress=None):
    """A filled label above (x, y), with an optional progress bar along its bottom."""
    font, scale = cv2.FONT_HERSHEY_SIMPLEX, 0.55
    (tw, th), _ = cv2.getTextSize(text, font, scale, 1)
    pad, h, w = 6, th + 12, tw + 12
    x = int(min(max(0, x), img.shape[1] - w))
    y = int(max(0, y - h - 4))
    cv2.rectangle(img, (x, y), (x + w, y + h), color, -1)
    if progress is not None:
        cv2.rectangle(img, (x, y + h - 3), (x + int(w * progress), y + h), (255, 255, 255), -1)
    cv2.putText(img, text, (x + pad, y + pad + th), font, scale, (20, 20, 20), 1, cv2.LINE_AA)
