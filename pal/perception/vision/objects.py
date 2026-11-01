"""Object recognition over time: what the detector finds, followed across frames and
recognized from earlier sightings.

Each tracked object is counted while in view. After FRAMES_TO_IDENTIFY frames its crop is
embedded once and matched against stored objects: above the match threshold it's recognized
as that object, otherwise it's stored as something new. The agent reads what's in view and
names or describes stored objects through `Objects`.
"""
import logging
import time
import zlib

import numpy as np

from pal.debug.preview import preview
from pal.perception.vision import draw, stored
from pal.perception.vision.draw import IDENTIFYING, PALETTE, RECOGNIZED, STORED
from pal.perception.vision.embed import crop, embed

logger = logging.getLogger(__name__)

FRAMES_TO_IDENTIFY = 30  # frames an object is followed before it's embedded and matched
MISS_GRACE = 10          # frames an object may drop out before its count starts over
SMOOTHING = 0.5          # weight of the newest box position; lower is steadier but lags behind movement
IN_VIEW_SECONDS = 2.0    # an object seen this recently counts as in view now
FLASH_SECONDS = 2.0      # how long a store or recognition stays highlighted


def _track(r, box) -> str | None:
    """The class with the track id, e.g. `keys#3`; None until the tracker confirms it."""
    return None if box.id is None else f"{r.names[int(box.cls)]}#{int(box.id)}"


class Objects:
    def __init__(self):
        self.frames = {}      # track -> frames in view
        self.missed = {}      # track -> frames in a row it wasn't seen
        self.smoothed = {}    # track id -> its last smoothed box
        self.identities = {}  # track -> the key of the stored object it was recognized as
        self.flashes = {}     # track -> (STORED or RECOGNIZED, when)
        self.seen_at = {}     # track -> (class, when it was last in view)
        # key -> stored entry without its vector: what's shown, and what the agent reads and changes
        self.known = {e["key"]: stored.without_vector(e) for e in stored.load()}

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
            self.seen_at[track] = (r.names[int(box.cls)], time.time())
            # once per track: one that's already identified isn't embedded again
            if self.frames[track] == FRAMES_TO_IDENTIFY and track not in self.identities:
                self._identify(r, box, track)
        for track in self.frames:
            self.missed[track] = 0 if track in seen else self.missed.get(track, 0) + 1
            if self.missed[track] > MISS_GRACE:
                self.frames[track] = 0

    def in_view(self) -> str:
        """What's in view now, for the agent: stored objects by key, with name and description,
        and what's still being identified. Empty when nothing is."""
        now = time.time()
        current = [(track, label) for track, (label, when) in list(self.seen_at.items()) if now - when <= IN_VIEW_SECONDS]
        parts, unidentified = [], {}
        for track, label in current:
            key = self.identities.get(track)
            if key is None:
                unidentified[label] = unidentified.get(label, 0) + 1
                continue
            entry = self.known[key]
            name = f"'{entry['name']}' ({label})" if entry.get("name") else f"a {label}, no name yet"
            parts.append(f"[{key}] {name}" + (f": {entry['description']}" if entry.get("description") else ""))
        parts += [f"{n} {label}{'s' if n > 1 else ''} not identified yet" for label, n in unidentified.items()]
        return "; ".join(parts)

    def update_object(self, key: str, name: str | None, description: str | None) -> bool:
        """Name or describe a stored object. False if nothing is stored under `key`."""
        entry = stored.update(key, name, description)
        if entry is None:
            return False
        self.known[key] = entry
        return True

    def shown(self, key: str) -> str:
        """How a stored object is labeled on screen: its name, or its class and key."""
        entry = self.known[key]
        return entry.get("name") or f"{entry['label']} {key}"

    def render(self, r) -> np.ndarray:
        """The frame with each object's mask, outline and label."""
        now = time.time()
        return draw.render(r, [self._style(r, box, now) for box in r.boxes])

    def _style(self, r, box, now):
        """(colour, label, progress or None, flashing) for one object."""
        track, label = _track(r, box), r.names[int(box.cls)]
        key = self.identities.get(track)
        kind, when = self.flashes.get(track, (None, 0))
        if now - when < FLASH_SECONDS:
            verb = "stored" if kind is STORED else "recognized"
            return kind, f"{verb}: {self.shown(key)}", None, True
        if key is not None:
            return PALETTE[zlib.crc32(key.encode()) % len(PALETTE)], self.shown(key), None, False
        progress = min(1.0, self.frames.get(track, 0) / FRAMES_TO_IDENTIFY) if track else 0.0
        return IDENTIFYING, label, progress, False

    def _identify(self, r, box, track):
        image = crop(r.orig_img, box)
        vector = embed([image])[0]
        key, sim = stored.best_match(vector, label=r.names[int(box.cls)])
        if key is not None:
            self.identities[track] = key
            self.flashes[track] = (RECOGNIZED, time.time())
            logger.info(f"recognized {track} as {self.shown(key)} [{key}] ({sim:.2f})")
            preview.show_object("recognized", image, self.shown(key), sim)
        else:
            entry = stored.store(track, vector, image)  # something new: it becomes a stored object of its own
            key = entry["key"]
            self.known[key] = entry
            self.identities[track] = key
            self.flashes[track] = (STORED, time.time())
            logger.info(f"stored {track} as new: {key} (closest {sim:.2f})")
            preview.show_object("stored", image, self.shown(key), sim)

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
