"""Object detection: YOLOE on each frame, with tracking so an object keeps its id across frames."""
import os
from pathlib import Path

# Ultralytics installs missing packages on its own; fail instead, so new dependencies get reviewed first
os.environ.setdefault("YOLO_AUTOINSTALL", "false")

from pal.config import VISION_MODELS

# What the detector looks for. YOLOE takes any words, not a fixed class list; keep it
# short, since many similar words mean more mix-ups.
CLASSES = [
    "person", "cat", "dog",                                    # who's around
    "keys", "phone", "wallet", "glasses", "headphones", "bag",  # what gets lost
    "laptop", "remote control", "bicycle",
]

# Low on purpose: the tracker needs weak detections to carry tracks through dips, and
# TRACKER's thresholds decide what counts.
DETECT_CONF = 0.1
TRACKER = Path(__file__).parent / "tracker.yaml"

_MODEL = None
_DEVICE = False  # False until decided; then "mps" or None (ultralytics finds CUDA itself)


def device() -> str | None:
    """The Apple GPU when there is one, for YOLOE and DINOv2 alike."""
    global _DEVICE
    if _DEVICE is False:
        import torch
        _DEVICE = "mps" if torch.backends.mps.is_available() else None
    return _DEVICE


def model():
    """Load YOLOE on first use — running detector-free should cost nothing."""
    global _MODEL
    if _MODEL is None:
        from ultralytics import YOLOE
        _MODEL = YOLOE(str(VISION_MODELS / "yoloe-26m-seg.pt"), verbose=False)
        _MODEL.set_classes(CLASSES)  # words to classes, through the text model (mobileclip2_b.ts)
    return _MODEL


def detect(frame):
    """YOLOE plus the tracker on one frame; the tracker keeps its state between calls."""
    return model().track(frame, persist=True, conf=DETECT_CONF, tracker=str(TRACKER),
                         verbose=False, device=device())[0]
