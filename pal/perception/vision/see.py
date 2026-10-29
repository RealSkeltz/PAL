import logging
import os
import queue
import time
from collections import deque

os.environ["OPENCV_LOG_LEVEL"] = "SILENT"

import cv2
import numpy as np

from typing import Generator, Iterable
from pal.config import VISION_MODELS

from pal.debug.preview import preview
from pal.debug.quiet import suppress_stderr

logger = logging.getLogger(__name__)

TARGET_FPS = 10
DELAY = int(1000 / TARGET_FPS)

# The last few seconds of frames, so a look can go back to when the user started
# talking: by the time the gate has decided, what they asked about may be gone.
BUFFER_SECONDS = 6
# Frames sent per look, spread from the start of the utterance to now. One sends
# just the newest frame.
FRAMES_PER_LOOK = 3

# Detector crops are square thumbnails; a whole frame is much larger, so cap its
# long edge before it becomes base64 in a prompt.
MAX_FRAME_EDGE = 768

_MODEL = None
_DEVICE = None  # Apple GPU when there is one; ultralytics finds CUDA itself


def _model():
    """Load YOLO on first use — running detector-free should cost nothing."""
    global _MODEL, _DEVICE
    if _MODEL is None:
        import torch
        from ultralytics import YOLO
        _DEVICE = "mps" if torch.backends.mps.is_available() else None
        _MODEL = YOLO(str(VISION_MODELS / "yolo26m.pt"), verbose=False)
    return _MODEL


def _find_closest_box(r, cx, cy):
    closest_box = None
    closest_dist = float('inf')
    for box in r.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        dist = (((x1 + x2) // 2 - cx) ** 2 + ((y1 + y2) // 2 - cy) ** 2) ** 0.5
        if dist < closest_dist:
            closest_dist = dist
            closest_box = box
    return closest_box

def _annotate_frame(r, closest_box):
    annotated = r.plot()
    h, w = annotated.shape[:2]
    cx, cy = w // 2, h // 2
    img = annotated.copy()

    if closest_box is not None:
        x1, y1, x2, y2 = map(int, closest_box.xyxy[0].tolist())
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)

    cv2.drawMarker(img, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
    #cv2.imshow("detections", img)
    return img

def _process_frame(frame, PREVIEW_MODE=False):
    results = _model()(frame, conf=0.375, verbose=False, device=_DEVICE)
    r = results[0]
    h, w = r.orig_img.shape[:2]
    closest_box = _find_closest_box(r, w // 2, h // 2)
    preview.set_frame(_annotate_frame(r, closest_box))
    return closest_box, r

def extract_crop(r, closest_box) -> tuple[np.ndarray, str, float]:
    x1, y1, x2, y2 = map(int, closest_box.xyxy[0].tolist())
    label = r.names[int(closest_box.cls)]
    conf = float(closest_box.conf)
    crop = cv2.resize(r.orig_img[y1:y2, x1:x2], (224, 224))
    return (crop, label, conf)



def downscale(frame: np.ndarray) -> np.ndarray:
    """Shrink a frame so its long edge is at most MAX_FRAME_EDGE."""
    h, w = frame.shape[:2]
    scale = MAX_FRAME_EDGE / max(h, w)
    if scale >= 1:
        return frame
    return cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def camera() -> Generator[np.ndarray, None, None]:
    """BGR frames from the laptop webcam, at about TARGET_FPS."""
    # Only the camera handshake is noisy — suppressing wider than this would
    # redirect fd 2 process-wide for the life of the generator, hiding every
    # thread's tracebacks.
    with suppress_stderr():
        cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield frame
        cv2.waitKey(DELAY)


def pick(buffer: deque, since: float, count: int) -> list[tuple[float, np.ndarray]]:
    """`count` (time, frame) pairs spread from `since` to the newest frame, oldest
    first. Fewer when the buffer is too short to give distinct frames."""
    newest = buffer[-1][0]
    start = min(max(since, buffer[0][0]), newest)
    if count == 1:
        targets = [newest]
    else:
        targets = [start + (newest - start) * i / (count - 1) for i in range(count)]
    picks = []
    for target in targets:
        closest = min(buffer, key=lambda entry: abs(entry[0] - target))
        if not picks or picks[-1] is not closest:
            picks.append(closest)
    return picks


def run(requests: queue.Queue, detect: bool = True,
        frames: Iterable[np.ndarray] | None = None) -> Generator[list[tuple[np.ndarray, str, float, float]], None, None]:
    """Capture on request and yield a list of (image, label, confidence, taken_at).

    A request is the time (epoch seconds) the question started. Without `detect`
    a look yields FRAMES_PER_LOOK whole frames from the buffer, spread from then
    to now — useful when the subject is the scene rather than one object, or when
    YOLO's classes don't cover what you're looking at. With `detect`, YOLO runs
    every frame and a look yields a crop of the object nearest the centre, now.

    `frames` replaces the webcam, e.g. with a remote device's camera; it sets the pace.
    """
    logger.info(f"capture loop started (detector {'on' if detect else 'off'})")
    buffer = deque()  # (time, downscaled frame), oldest first

    for frame in camera() if frames is None else frames:
        now = time.time()
        buffer.append((now, downscale(frame)))
        while buffer[0][0] < now - BUFFER_SECONDS:
            buffer.popleft()

        if detect:
            closest_box, r = _process_frame(frame)
        else:
            # No detector, so nothing annotates the preview — push the raw frame.
            preview.set_frame(frame)

        try:
            since = requests.get_nowait()
        except queue.Empty:
            continue

        capture = []
        if not detect:
            capture = [(image, "scene", 1.0, at) for at, image in pick(buffer, since, FRAMES_PER_LOOK)]
            ago = ", ".join(f"{now - at:.1f}s" for *_, at in capture)
            logger.info(f"looking: {len(capture)} frame(s) from {ago} ago")
            preview.show_sent([(image, now - at) for image, *_, at in capture])
        elif closest_box is None:
            logger.info("look requested but no object detected")
        else:
            label = r.names[int(closest_box.cls)]
            conf = float(closest_box.conf)
            logger.debug(f"look requested, capturing: {label} ({conf:.2f})")
            capture = [(*extract_crop(r, closest_box), now)]
        if capture:
            yield capture
