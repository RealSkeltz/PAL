"""The camera: a rolling buffer of frames for looks, and optionally object recognition.

A look sends whole frames from the buffer. With `detect`, YOLOE also runs on every
frame: objects are tracked across frames, and once one has been in view long enough
its crop is embedded (DINOv2) and matched against earlier sightings, so it's either
recognized or stored as something new. The preview shows all of it.

Run `python -m pal.perception.vision.see` to watch recognition on the webcam.
"""
import json
import logging
import os
import queue
import time
import zlib
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

os.environ["OPENCV_LOG_LEVEL"] = "SILENT"
# Ultralytics installs missing packages on its own; fail instead, so new dependencies get reviewed first
os.environ.setdefault("YOLO_AUTOINSTALL", "false")

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

# A whole frame is large, so cap its long edge before it becomes base64 in a prompt.
MAX_FRAME_EDGE = 768

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
DINO = VISION_MODELS / "dinov2-base"  # facebook/dinov2-base, saved locally so loading needs no network

FRAMES_TO_IDENTIFY = 30  # frames an object is followed before it's embedded and matched
MISS_GRACE = 10          # frames an object may drop out before its count starts over
SMOOTHING = 0.5          # weight of the newest box position; lower is steadier but lags behind movement
# A first guess: set it between your own same-object and different-object similarities
MATCH_THRESHOLD = 0.6

# Sightings, for now next to the code (gitignored): one entry per stored object
OBJECTS_FILE = Path(__file__).parent / "vectors.json"
CROPS_DIR = Path(__file__).parent / "crops"

_MODEL = None
_DEVICE = None  # Apple GPU when there is one; ultralytics finds CUDA itself
_DINO = None


def _model():
    """Load YOLOE on first use — running detector-free should cost nothing."""
    global _MODEL, _DEVICE
    if _MODEL is None:
        import torch
        from ultralytics import YOLOE
        _DEVICE = "mps" if torch.backends.mps.is_available() else None
        _MODEL = YOLOE(str(VISION_MODELS / "yoloe-26m-seg.pt"), verbose=False)
        _MODEL.set_classes(CLASSES)  # words to classes, through the text model (mobileclip2_b.ts)
    return _MODEL


def _detect(frame):
    """YOLOE plus the tracker on one frame; the tracker keeps its state between calls."""
    return _model().track(frame, persist=True, conf=DETECT_CONF, tracker=str(TRACKER),
                          verbose=False, device=_DEVICE)[0]


def _embedder():
    """Load DINOv2 on first use."""
    global _DINO
    if _DINO is None:
        from transformers import AutoImageProcessor, AutoModel
        _model()  # sets _DEVICE
        _DINO = (AutoImageProcessor.from_pretrained(DINO),
                 AutoModel.from_pretrained(DINO).to(_DEVICE or "cpu").eval())
    return _DINO


def _embed(crops: list[np.ndarray]) -> np.ndarray:
    """One L2-normalized vector per crop (BGR, as cropped from the frame). The dot product
    of two vectors is their cosine similarity."""
    import torch
    proc, model = _embedder()
    images = [cv2.cvtColor(c, cv2.COLOR_BGR2RGB) for c in crops]
    with torch.no_grad():
        out = model(**proc(images=images, return_tensors="pt").to(model.device))
    return torch.nn.functional.normalize(out.last_hidden_state[:, 0], dim=-1).cpu().numpy()


def _crop(img, box, pad=0.1):
    """The box's region of `img`, with `pad` (fraction of the box size) around it."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box.xyxy[0].tolist()
    dx, dy = (x2 - x1) * pad, (y2 - y1) * pad
    x1, y1 = max(0, int(x1 - dx)), max(0, int(y1 - dy))
    x2, y2 = min(w, int(x2 + dx)), min(h, int(y2 + dy))
    return img[y1:y2, x1:x2].copy()


def _track(r, box) -> str | None:
    """The class with the track id, e.g. `keys#3`; None until the tracker confirms it."""
    return None if box.id is None else f"{r.names[int(box.cls)]}#{int(box.id)}"


def _sightings() -> list[dict]:
    return json.loads(OBJECTS_FILE.read_text()) if OBJECTS_FILE.exists() else []


def _best_match(vector, label=None, threshold=MATCH_THRESHOLD) -> tuple[str | None, float]:
    """What `vector` is recognized as: the most similar stored object's name (or its track
    while unnamed), and the similarity. The name is None when nothing reaches `threshold`.
    With `label`, only objects of that class count."""
    entries = [e for e in _sightings() if label is None or e["label"] == label]
    if not entries:
        return None, 0.0
    saved = np.array([e["vector"] for e in entries], dtype=np.float32)
    sims = saved @ vector / (np.linalg.norm(saved, axis=1) * np.linalg.norm(vector))
    best = int(np.argmax(sims))
    if sims[best] < threshold:
        return None, float(sims[best])
    return entries[best].get("name") or entries[best]["track"], float(sims[best])


def _store(track: str, vector: np.ndarray, crop: np.ndarray):
    """Append one object to OBJECTS_FILE: class, track, time, the crop's file and the vector."""
    now = datetime.now(timezone.utc)
    CROPS_DIR.mkdir(exist_ok=True)
    crop_path = CROPS_DIR / f"{track.replace('#', '_')}_{now:%Y%m%d-%H%M%S}.jpg"
    cv2.imwrite(str(crop_path), crop)
    entries = _sightings()
    entries.append({
        "label": track.split("#")[0],
        "track": track,
        "name": None,  # filled in later from what the user says, e.g. "my keys"
        "at": now.isoformat(),
        "crop": crop_path.name,
        "vector": [round(float(x), 5) for x in vector],
    })
    OBJECTS_FILE.write_text(json.dumps(entries))


# --- Drawing -----------------------------------------------------------------

# BGR, the preview page's colours
IDENTIFYING = (150, 150, 150)
STORED = (112, 158, 25)       # green
RECOGNIZED = (229, 135, 57)   # blue
PALETTE = [(0, 133, 201), (129, 81, 213), (233, 133, 144), (38, 89, 217), (180, 160, 40)]
FLASH_SECONDS = 2.0           # how long a store or recognition stays highlighted


def _outline(r, i) -> list[np.ndarray]:
    """The contours of object `i`'s mask, in frame coordinates; empty without a mask."""
    if r.masks is None:
        return []
    h, w = r.orig_img.shape[:2]
    mask = cv2.resize(r.masks.data[i].cpu().numpy(), (w, h)) > 0.5
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c for c in contours if cv2.contourArea(c) > 0.002 * h * w]  # drop specks


def _pill(img, text, x, y, color, progress=None):
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


class Objects:
    """What the detector finds, followed across frames and recognized from earlier sightings.

    Each tracked object is counted while in view. After FRAMES_TO_IDENTIFY frames its crop is
    embedded once and matched against stored objects: above MATCH_THRESHOLD it's recognized as
    that object, otherwise it's stored as something new and becomes its own identity.
    """

    def __init__(self):
        self.frames = {}      # track -> frames in view
        self.missed = {}      # track -> frames in a row it wasn't seen
        self.smoothed = {}    # track id -> its last smoothed box
        self.identities = {}  # track -> what it was recognized as, e.g. "keys#3" or later a name
        self.flashes = {}     # track -> (STORED or RECOGNIZED, when)

    def update(self, r):
        """Take in one tracked frame: steady the boxes, count, and identify what's due."""
        self._smooth(r)
        seen = set()
        for box in r.boxes:
            track = _track(r, box)
            if track is None:
                continue
            seen.add(track)
            self.frames[track] = self.frames.get(track, 0) + 1
            # once per track: one that's already identified isn't embedded again
            if self.frames[track] == FRAMES_TO_IDENTIFY and track not in self.identities:
                self._identify(r, box, track)
        for track in self.frames:
            self.missed[track] = 0 if track in seen else self.missed.get(track, 0) + 1
            if self.missed[track] > MISS_GRACE:
                self.frames[track] = 0

    def render(self, r) -> np.ndarray:
        """The frame with each object's mask, outline and label."""
        img = r.orig_img.copy()
        if not len(r.boxes):
            return img
        now = time.time()
        styled = [(box, _outline(r, i), *self._style(r, box, now)) for i, box in enumerate(r.boxes)]

        fill = img.copy()
        for _, contours, color, *_ in styled:
            cv2.drawContours(fill, contours, -1, color, -1)
        img = cv2.addWeighted(fill, 0.3, img, 0.7, 0)

        for box, contours, color, text, progress, flashing in styled:
            cv2.drawContours(img, contours, -1, color, 4 if flashing else 2, cv2.LINE_AA)
            if contours:  # the label sits on the object's highest point
                top = min((tuple(p[0]) for c in contours for p in c), key=lambda p: p[1])
                _pill(img, text, top[0] - 40, top[1], color, progress)
            else:
                x1, y1 = box.xyxy[0][:2].tolist()
                _pill(img, text, x1, y1, color, progress)
        return img

    def _style(self, r, box, now):
        """(colour, label, progress or None, flashing) for one object."""
        track, label = _track(r, box), r.names[int(box.cls)]
        identity = self.identities.get(track)
        kind, when = self.flashes.get(track, (None, 0))
        if now - when < FLASH_SECONDS:
            verb = "stored" if kind is STORED else "recognized"
            return kind, f"{verb}: {identity}", None, True
        if identity is not None:
            return PALETTE[zlib.crc32(identity.encode()) % len(PALETTE)], identity, None, False
        progress = min(1.0, self.frames.get(track, 0) / FRAMES_TO_IDENTIFY) if track else 0.0
        return IDENTIFYING, label, progress, False

    def _identify(self, r, box, track):
        crop = _crop(r.orig_img, box)
        vector = _embed([crop])[0]
        match, sim = _best_match(vector, label=r.names[int(box.cls)])
        if match is not None:
            self.identities[track] = match
            self.flashes[track] = (RECOGNIZED, time.time())
            logger.info(f"recognized {track} as {match} ({sim:.2f})")
            preview.show_object("recognized", crop, match, sim)
        else:
            _store(track, vector, crop)  # something new: it becomes its own identity
            self.identities[track] = track
            self.flashes[track] = (STORED, time.time())
            logger.info(f"stored {track} as new (closest {sim:.2f})")
            preview.show_object("stored", crop, track, sim)

    def _smooth(self, r):
        """Steady the boxes: each tracked box becomes a running average of where it has been."""
        if r.boxes.id is None:
            return
        data = r.boxes.data.clone()  # the result's own tensor can't be changed in place
        for i, track_id in enumerate(r.boxes.id.int().tolist()):
            if track_id in self.smoothed:
                data[i, :4] = SMOOTHING * data[i, :4] + (1 - SMOOTHING) * self.smoothed[track_id]
            self.smoothed[track_id] = data[i, :4].clone()
        r.update(boxes=data)


# --- Frames and looks --------------------------------------------------------

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

    A request is the time (epoch seconds) the question started. A look yields
    FRAMES_PER_LOOK whole frames from the buffer, spread from then to now. With
    `detect`, YOLOE also runs on every frame to follow and recognize objects
    (see `Objects`); the preview shows what it found.

    `frames` replaces the webcam, e.g. with a remote device's camera; it sets the pace.
    """
    logger.info(f"capture loop started (detector {'on' if detect else 'off'})")
    buffer = deque()  # (time, downscaled frame), oldest first
    objects = Objects() if detect else None

    for frame in camera() if frames is None else frames:
        now = time.time()
        buffer.append((now, downscale(frame)))
        while buffer[0][0] < now - BUFFER_SECONDS:
            buffer.popleft()

        if detect:
            r = _detect(frame)
            objects.update(r)
            preview.set_frame(objects.render(r))
        else:
            preview.set_frame(frame)

        try:
            since = requests.get_nowait()
        except queue.Empty:
            continue

        capture = [(image, "scene", 1.0, at) for at, image in pick(buffer, since, FRAMES_PER_LOOK)]
        ago = ", ".join(f"{now - at:.1f}s" for *_, at in capture)
        logger.info(f"looking: {len(capture)} frame(s) from {ago} ago")
        preview.show_sent([(image, now - at) for image, *_, at in capture])
        yield capture


def main():
    """Watch detection and recognition on the webcam, in a window and in the preview. q quits."""
    import webbrowser
    from pal.log import setup
    setup()
    preview.start(port=4200)
    webbrowser.open("http://127.0.0.1:4200")
    objects = Objects()
    for frame in camera():
        r = _detect(frame)
        objects.update(r)
        view = objects.render(r)
        preview.set_frame(view)
        cv2.imshow("pal: objects", view)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
