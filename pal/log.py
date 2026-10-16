"""Logging and session recording. Call `setup()` once at startup.

Every module logs through its own `logging.getLogger(__name__)`. Each run
gets a folder in data/logs/ holding:

    pal.log         every log line from PAL, DEBUG and up, with timestamps
    session.jsonl   what PAL took in and decided, one JSON object per line

The terminal (and the preview, when it runs) shows INFO and up. Other
libraries only get through at WARNING, so their chatter stays out.

The session file is for looking back and replaying: every event and camera
capture as it arrived, every gate decision, and every turn in its final
state. `read_session` turns it back into objects.
"""
import json
import logging
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

from pydantic import TypeAdapter

from pal.config import LOGS_DIR
from pal.schemas.event import event_adapter
from pal.schemas.turn import ConversationTurn

# Libraries whose warnings are known noise. phonemizer (inside Kokoro) warns
# about word-count mismatches on most sentences; the speech is fine.
QUIET_LOGGERS = ("phonemizer",)

_session_file = None
_session_lock = threading.Lock()
_adapters: dict[type, TypeAdapter] = {}
_turn_adapter: TypeAdapter = TypeAdapter(ConversationTurn)


class _ShortNameFormatter(logging.Formatter):
    """Adds %(short)s: the last part of the logger name, e.g. "router"."""

    def format(self, record):
        record.short = record.name.rsplit(".", 1)[-1]
        return super().format(record)


CONSOLE_FORMAT = _ShortNameFormatter("%(asctime)s %(short)-10s %(message)s", datefmt="%H:%M:%S")
FILE_FORMAT = _ShortNameFormatter("%(asctime)s.%(msecs)03d %(levelname)-7s %(name)s: %(message)s", datefmt="%H:%M:%S")


def setup(verbose: bool = False) -> Path:
    """Configure logging for this run and return its log folder."""
    global _session_file
    run_dir = LOGS_DIR / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG if verbose else logging.INFO)
    console.setFormatter(CONSOLE_FORMAT)

    file = logging.FileHandler(run_dir / "pal.log")
    file.setLevel(logging.DEBUG)
    file.setFormatter(FILE_FORMAT)

    from pal.debug.preview import PreviewHandler
    preview = PreviewHandler()
    preview.setLevel(logging.INFO)
    preview.setFormatter(CONSOLE_FORMAT)

    root = logging.getLogger()
    root.setLevel(logging.WARNING)  # third-party libraries
    for handler in (console, file, preview):
        root.addHandler(handler)
    for ours in ("pal", "agents"):
        logging.getLogger(ours).setLevel(logging.DEBUG)
    for noisy in QUIET_LOGGERS:
        logging.getLogger(noisy).setLevel(logging.ERROR)

    threading.excepthook = _log_thread_crash
    _session_file = open(run_dir / "session.jsonl", "a", buffering=1)
    logging.getLogger(__name__).info(f"logging to {run_dir}")
    return run_dir


def record(kind: str, data: Any):
    """Append one entry to the session file. Does nothing before `setup()`."""
    if _session_file is None:
        return
    line = json.dumps({"t": time.time(), "kind": kind, "data": _dump(data)}, default=str)
    with _session_lock:
        _session_file.write(line + "\n")


def read_session(path: Path) -> Iterator[tuple[float, str, Any]]:
    """Yield (time, kind, data), with events and turns rebuilt as objects."""
    for line in Path(path).read_text().splitlines():
        entry = json.loads(line)
        kind, data = entry["kind"], entry["data"]
        if kind in ("event", "capture"):
            data = event_adapter.validate_python(data)
        elif kind == "turn":
            data = _turn_adapter.validate_python(data)
        yield entry["t"], kind, data


def _dump(data: Any) -> Any:
    if isinstance(data, dict):
        return data
    adapter = _adapters.setdefault(type(data), TypeAdapter(type(data)))
    return adapter.dump_python(data, mode="json")


def _log_thread_crash(args):
    logging.getLogger("pal").error(
        f"uncaught exception in thread {args.thread.name if args.thread else '?'}",
        exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
    )
