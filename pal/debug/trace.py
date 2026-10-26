"""When each stage of handling one utterance happened, for the preview's latency bar.

Stages are marked as they pass, keyed by the utterance's event id: listen marks
when the speech ended and was transcribed, the router when the gate started and
finished, the responder takes the marks for its own timing line. Debug only:
nothing decides anything from these.
"""
import threading
import time
from uuid import UUID

_marks: dict[UUID, dict[str, float]] = {}
_lock = threading.Lock()
KEEP = 50  # utterances never answered (said to someone else) would otherwise pile up


def mark(key: UUID, stage: str, at: float | None = None):
    with _lock:
        _marks.setdefault(key, {})[stage] = time.time() if at is None else at
        while len(_marks) > KEEP:
            _marks.pop(next(iter(_marks)))


def take(key: UUID) -> dict[str, float]:
    with _lock:
        return _marks.pop(key, {})
