import os
import threading

os.environ["OPENCV_LOG_LEVEL"] = "SILENT"

import cv2
import numpy as np

from typing import Generator
from pal.config import VISION_MODELS
from pal.voice.sounds import capture_tone
from pal.types import Message
from pal.voice.speak import play_sound

from pal.debug.preview import preview

from pal.controls.headset import start_headset_listener

TARGET_FPS = 30
DELAY = int(1000 / TARGET_FPS)

# Detector crops are square thumbnails; a whole frame is much larger, so cap its
# long edge before it becomes base64 in a prompt.
MAX_FRAME_EDGE = 768

_MODEL = None


def _model():
    """Load YOLO on first use — running detector-free should cost nothing."""
    global _MODEL
    if _MODEL is None:
        from ultralytics import YOLO
        _MODEL = YOLO(str(VISION_MODELS / "yolov8n.pt"), verbose=False)
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
        label = r.names[int(closest_box.cls)]
        conf = float(closest_box.conf)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(img, f"{label} {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.drawMarker(img, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
    #cv2.imshow("detections", img)
    return img

def _process_frame(frame, PREVIEW_MODE=False):
    results = _model()(frame, conf=0.375, verbose=False)
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


def run(trigger: threading.Event, detect: bool = True) -> Generator[Message, None, None]:
    """Capture on trigger and yield (image, label, confidence).

    With `detect`, YOLO runs every frame and the trigger yields a crop of the
    object nearest the centre. Without it, no detector runs at all and the
    trigger yields the whole frame — useful when the subject is the scene rather
    than one object, or when YOLO's classes don't cover what you're looking at.
    """
    # Only the camera handshake is noisy — suppressing wider than this would
    # redirect fd 2 process-wide for the life of the generator, hiding every
    # thread's tracebacks.
    with suppress_stderr():
        cap = cv2.VideoCapture(0)

    print(f"[Vision] Capture loop started (detector {'on' if detect else 'off'})")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if detect:
            closest_box, r = _process_frame(frame)
        else:
            # No detector, so nothing annotates the preview — push the raw frame.
            preview.set_frame(frame)

        if trigger.is_set():
            capture = None
            if not detect:
                print("[Vision] Trigger fired, capturing full frame")
                capture = (downscale(frame), "scene", 1.0)
            elif closest_box is None:
                print("[Vision] Trigger fired but no object detected")
            else:
                label = r.names[int(closest_box.cls)]
                conf = float(closest_box.conf)
                print(f"[Vision] Trigger fired, capturing: {label} ({conf:.2f})")
                capture = extract_crop(r, closest_box)

            # Release the trigger before yielding — the capture already happened,
            # so a slow consumer shouldn't hold the camera armed.
            trigger.clear()
            if capture is not None:
                play_sound(capture_tone())
                yield capture

        cv2.waitKey(DELAY)


# Warning wrapper
import contextlib

@contextlib.contextmanager
def suppress_stderr():
    with open(os.devnull, 'w') as devnull:
        old_stderr = os.dup(2)
        os.dup2(devnull.fileno(), 2)
        try:
            yield
        finally:
            os.dup2(old_stderr, 2)
            os.close(old_stderr)