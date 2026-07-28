"""Live view of what the agent is doing, for the HUD to render.

Written from the audio, agent, and interaction threads; read by the vision
thread once per frame. Every access goes through a lock and readers get an
immutable snapshot, so a frame can never catch a half-updated state.

Tool activity is deliberately not a State. A tool runs *while* the agent is
thinking, so it rides alongside as a transient badge rather than replacing the
state and hiding what the agent was already doing.
"""

import threading
import time
from dataclasses import dataclass
from enum import Enum


class State(str, Enum):
    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"


# Labels passed to `timed()` that are worth a row on the HUD, in display order.
# Anything else recorded is kept but not shown — the panel stays legible only if
# it refuses to grow.
SHOWN_TIMINGS = (("STT", "stt"), ("reply", "reply"))

TOOL_PREFIX = "tool_"
TOOL_BADGE_SECONDS = 3.0


@dataclass(frozen=True)
class Snapshot:
    state: State
    timings: tuple[tuple[str, float], ...]   # (display label, milliseconds)
    tool: str | None                         # recently fired tool, if any
    tool_age: float                          # seconds since it fired


class Status:
    def __init__(self):
        self._lock = threading.Lock()
        self._state = State.IDLE
        self._timings: dict[str, float] = {}
        self._last_tool_key: str | None = None
        self._tool: str | None = None
        self._tool_at = 0.0

    def set_state(self, state: State):
        with self._lock:
            self._state = state

    def record(self, label: str, milliseconds: float):
        """Called for every `timed()` block, so new instrumentation shows up free."""
        with self._lock:
            self._timings[label] = milliseconds
            if label.startswith(TOOL_PREFIX):
                self._last_tool_key = label

    def tool_fired(self, name: str):
        with self._lock:
            self._tool = name
            self._tool_at = time.monotonic()

    def snapshot(self) -> Snapshot:
        with self._lock:
            state = self._state
            timings = dict(self._timings)
            last_tool_key = self._last_tool_key
            tool, tool_at = self._tool, self._tool_at

        age = time.monotonic() - tool_at if tool else 0.0
        if tool and age > TOOL_BADGE_SECONDS:
            tool = None

        return Snapshot(state, _rows(timings, last_tool_key), tool, age)


def _rows(timings, last_tool_key):
    rows = [(label, timings[key]) for key, label in SHOWN_TIMINGS if key in timings]
    if last_tool_key in timings:
        rows.append((last_tool_key[len(TOOL_PREFIX):], timings[last_tool_key]))
    return tuple(rows)


# Module-level singleton, mirroring how `preview` is used.
status = Status()
