"""Tools every Pal gets: notes, timers, saving the last captured frame."""
import json
import time
import threading

import cv2

from pal.config import NOTES_FILE, SAVED_IMAGES_DIR
from pal.tools.base import ToolHandler, tool


class BuiltinTools(ToolHandler):
    def __init__(self, pal):
        super().__init__(pal)
        self._load_notes()

    def _load_notes(self):
        if NOTES_FILE.exists():
            self.notes = json.loads(NOTES_FILE.read_text())
        else:
            self.notes = []

    def _save_notes(self):
        NOTES_FILE.parent.mkdir(parents=True, exist_ok=True)
        NOTES_FILE.write_text(json.dumps(self.notes, indent=2))

    @tool(
        "Save a note for the user to refer back to later.",
        params={"content": {"type": "string"}},
        required=["content"],
    )
    def save_note(self, content: str) -> str:
        self.notes.append({"content": content, "timestamp": time.time()})
        self._save_notes()
        return f"Note saved: {content}"

    @tool("Retrieve previously saved notes.")
    def list_notes(self) -> str:
        if not self.notes:
            return "No notes saved."
        return "\n".join(f"{i}. {n['content']}" for i, n in enumerate(self.notes, 1))

    @tool(
        "Save the most recently captured image to disk, optionally with a note about it.",
        params={"filename": {"type": "string"}, "note": {"type": "string"}},
    )
    def save_last_image(self, filename: str = None, note: str = None) -> str:
        if self.pal.last_image is None:
            return "No image available to save."
        if filename is None:
            filename = f"capture_{int(time.time())}"
        SAVED_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(SAVED_IMAGES_DIR / f"{filename}.jpg"), self.pal.last_image)
        if note is None:
            return f"Image saved as {filename}.jpg"
        # JPEG has no convenient place for text via cv2, so the note lives in a sidecar file.
        sidecar = {"image": f"{filename}.jpg", "note": note, "timestamp": time.time()}
        (SAVED_IMAGES_DIR / f"{filename}.json").write_text(json.dumps(sidecar, indent=2))
        return f"Image saved as {filename}.jpg with note: {note}"

    @tool(
        "Set a timer that will alert the user after a duration in seconds.",
        params={"seconds": {"type": "integer"}, "label": {"type": "string"}},
        required=["seconds"],
    )
    def set_timer(self, seconds: int, label: str = "timer") -> str:
        def fire():
            time.sleep(seconds)
            print(f"[Timer] {label} done!")
            # Could also queue a system message into the conversation

        threading.Thread(target=fire, daemon=True).start()
        return f"Timer set for {seconds} seconds: {label}"
