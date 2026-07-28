"""Temporal smoothing for detections.

YOLO runs independently on every frame, so the raw stream wobbles: corners drift
a few pixels, boxes strobe as confidence crosses the threshold, and labels
occasionally flip. This turns that into stable tracks — ByteTrack supplies the
identities, an EMA settles the corners, and hysteresis carries a box through a
short dropout instead of letting it vanish.
"""

from dataclasses import dataclass


DETECT_FLOOR = 0.1
CREATE_CONF = 0.375
SMOOTHING = 0.4
MIN_HITS = 3
MAX_MISSES = 30
FADE_FRAMES = 6


@dataclass
class Track:
    """One detected object, smoothed over time."""

    id: int
    label: str
    conf: float
    xyxy: tuple[float, float, float, float]
    hits: int = 1
    misses: int = 0
    best_conf: float = 0.0

    @property
    def opacity(self) -> float:
        return max(0.0, 1.0 - self.misses / FADE_FRAMES)

    @property
    def visible(self) -> bool:
        return self.hits >= MIN_HITS and self.opacity > 0


class Tracker:
    """Turns per-frame YOLO results into a stable set of Tracks."""

    def __init__(self):
        self._tracks: dict[int, Track] = {}

    def update(self, result) -> list[Track]:
        """Fold one frame's detections in and return the tracks worth drawing."""
        self._absorb(result)

        for track in list(self._tracks.values()):
            if track.misses > MAX_MISSES:
                del self._tracks[track.id]

        return [t for t in self._tracks.values() if t.visible]

    def reset(self):
        self._tracks.clear()

    # ----- internals -----

    def _absorb(self, result):
        seen: set[int] = set()

        for track_id, label, conf, box in _detections(result):
            existing = self._tracks.get(track_id)
            if existing is None:
                if conf >= CREATE_CONF:
                    self._tracks[track_id] = Track(
                        id=track_id, label=label, conf=conf, xyxy=box, best_conf=conf
                    )
                    seen.add(track_id)
                continue

            seen.add(track_id)
            existing.xyxy = _ease(existing.xyxy, box)
            existing.conf = conf
            existing.hits += 1
            existing.misses = 0
            # Only let a more confident sighting rename the track. ByteTrack
            # associates on geometry alone and never looks at class, so the label
            # on a given ID can flip between frames.
            if conf > existing.best_conf:
                existing.best_conf = conf
                existing.label = label

        for track_id, track in self._tracks.items():
            if track_id not in seen:
                track.misses += 1


def _detections(result):
    """Yield (track_id, label, conf, xyxy) for each tracked box in a result."""
    boxes = getattr(result, "boxes", None)
    if boxes is None or boxes.id is None:
        return  # tracker assigned no identities this frame

    ids = boxes.id.int().tolist()
    classes = boxes.cls.int().tolist()
    confs = boxes.conf.tolist()
    coords = boxes.xyxy.tolist()

    for track_id, cls, conf, box in zip(ids, classes, confs, coords):
        yield track_id, result.names[cls], float(conf), tuple(box)


def _ease(previous, target):
    """Exponential moving average, corner by corner."""
    return tuple(p + SMOOTHING * (t - p) for p, t in zip(previous, target))
