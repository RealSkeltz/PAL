import contextlib
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
from pal.perception.vision.hud import annotate
from pal.perception.vision.tracking import DETECT_FLOOR, Tracker

TARGET_FPS = 30
DELAY = int(1000 / TARGET_FPS)

# Detector crops are square thumbnails; a whole frame is much larger, so cap its
# long edge before it becomes base64 in a prompt.
MAX_FRAME_EDGE = 768

_MODEL = None
_tracker = Tracker()


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


def _model():
    """Load YOLO on first use — running detector-free should cost nothing."""
    global _MODEL
    if _MODEL is None:
        from ultralytics import YOLO
        _MODEL = YOLO(str(VISION_MODELS / "yolov8n.pt"), verbose=False)
    return _MODEL


def _process_frame(frame):
    # Detect right down at DETECT_FLOOR and let the tracker decide what earns a
    # box — filtering higher here would starve ByteTrack's low-score recovery.
    result = _model().track(frame, conf=DETECT_FLOOR, persist=True,
                            tracker="bytetrack.yaml", verbose=False)[0]
    tracks = _tracker.update(result)
    preview.set_frame(annotate(frame, tracks))
    return result


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
            _process_frame(frame)
        else:
            # No detector, so nothing annotates the preview — push the raw frame.
            preview.set_frame(frame)

        if trigger.is_set():
            capture = None
            print("[Vision] Trigger fired, capturing full frame")
            capture = (downscale(frame), "scene", 1.0)

            # Release the trigger before yielding — the capture already happened,
            # so a slow consumer shouldn't hold the camera armed.
            trigger.clear()
            if capture is not None:
                play_sound(capture_tone())
                yield capture

        cv2.waitKey(DELAY)
