"""Stored objects: every person, animal or thing recognition has kept, with its vector.

For now in a file next to the code (gitignored): one entry per object, with a stable key
("o4") the agent names and describes it by, and the crop it was stored from.
"""
import json
import threading
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

# A first guess: set it between your own same-object and different-object similarities
MATCH_THRESHOLD = 0.6

OBJECTS_FILE = Path(__file__).parent / "vectors.json"
CROPS_DIR = Path(__file__).parent / "crops"
_lock = threading.Lock()  # OBJECTS_FILE is rewritten from the camera thread and by the agent


def load() -> list[dict]:
    """Stored objects, oldest first. One stored before objects had keys gets one here, the same
    every time, until the file is next written with it."""
    entries = json.loads(OBJECTS_FILE.read_text()) if OBJECTS_FILE.exists() else []
    n = max((int(e["key"][1:]) for e in entries if e.get("key")), default=0)
    for e in entries:
        if not e.get("key"):
            n += 1
            e["key"] = f"o{n}"
    return entries


def without_vector(entry: dict) -> dict:
    return {k: v for k, v in entry.items() if k != "vector"}


def best_match(vector, label=None, threshold=MATCH_THRESHOLD) -> tuple[str | None, float]:
    """What `vector` is recognized as: the key of the most similar stored object, and the
    similarity. The key is None when nothing reaches `threshold`. With `label`, only objects
    of that class count."""
    entries = [e for e in load() if label is None or e["label"] == label]
    if not entries:
        return None, 0.0
    saved = np.array([e["vector"] for e in entries], dtype=np.float32)
    sims = saved @ vector / (np.linalg.norm(saved, axis=1) * np.linalg.norm(vector))
    best = int(np.argmax(sims))
    if sims[best] < threshold:
        return None, float(sims[best])
    return entries[best]["key"], float(sims[best])


def store(track: str, vector: np.ndarray, crop: np.ndarray) -> dict:
    """Store a new object: its key, class, track, time, the crop's file and the vector.
    Returns the entry without its vector."""
    now = datetime.now(timezone.utc)
    CROPS_DIR.mkdir(exist_ok=True)
    crop_path = CROPS_DIR / f"{track.replace('#', '_')}_{now:%Y%m%d-%H%M%S}.jpg"
    cv2.imwrite(str(crop_path), crop)
    with _lock:
        entries = load()
        entry = {
            "key": f"o{max((int(e['key'][1:]) for e in entries), default=0) + 1}",
            "label": track.split("#")[0],
            "track": track,
            "name": None,         # given by the agent, e.g. "my keys" or "Anna"
            "description": None,  # likewise, e.g. "black leather keyring"
            "at": now.isoformat(),
            "crop": crop_path.name,
            "vector": [round(float(x), 5) for x in vector],
        }
        OBJECTS_FILE.write_text(json.dumps([*entries, entry]))
    return without_vector(entry)


def update(key: str, name: str | None, description: str | None) -> dict | None:
    """Set a stored object's name and/or description. The entry without its vector, or None if
    nothing is stored under `key`."""
    with _lock:
        entries = load()
        entry = next((e for e in entries if e["key"] == key), None)
        if entry is None:
            return None
        entry.update({k: v for k, v in (("name", name), ("description", description)) if v is not None})
        OBJECTS_FILE.write_text(json.dumps(entries))
    return without_vector(entry)
