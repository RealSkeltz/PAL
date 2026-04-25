import json
import time
import threading
from pathlib import Path
import cv2

NOTES_FILE = Path("/Users/jscheltema/Documents/Personal/PAL/Shared/Resources/notes.json")
SAVED_IMAGES_DIR = Path("/Users/jscheltema/Documents/Personal/PAL/Shared/Resources/saved_images")
SAVED_IMAGES_DIR.mkdir(parents=True, exist_ok=True)


# --- Tool specs (what gets sent to the model) ---

TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "save_note",
            "description": "Save a note for the user to refer back to later.",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {"type": "string"},
                },
                "required": ["content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_notes",
            "description": "Retrieve previously saved notes.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_last_image",
            "description": "Save the most recently captured image to disk.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_timer",
            "description": "Set a timer that will alert the user after a duration in seconds.",
            "parameters": {
                "type": "object",
                "properties": {
                    "seconds": {"type": "integer"},
                    "label": {"type": "string"},
                },
                "required": ["seconds"],
            },
        },
    },
    # search_web omitted from initial integration since it's a placeholder
]


# --- Handlers ---

class ToolHandler:
    def __init__(self, pal):
        self.pal = pal  # reference back so we can read last_image, etc.
        self._load_notes()
    
    def _load_notes(self):
        if NOTES_FILE.exists():
            self.notes = json.loads(NOTES_FILE.read_text())
        else:
            self.notes = []
    
    def _save_notes(self):
        NOTES_FILE.parent.mkdir(parents=True, exist_ok=True)
        NOTES_FILE.write_text(json.dumps(self.notes, indent=2))
    
    def dispatch(self, name: str, arguments: dict) -> str:
        """Execute a tool call and return a string result for the model."""
        handler = getattr(self, f"_tool_{name}", None)
        if handler is None:
            return f"Error: unknown tool '{name}'"
        try:
            return handler(**arguments)
        except Exception as e:
            return f"Error: {e}"
    
    def _tool_save_note(self, content: str) -> str:
        entry = {"content": content, "timestamp": time.time()}
        self.notes.append(entry)
        self._save_notes()
        return f"Note saved: {content}"
    
    def _tool_list_notes(self) -> str:
        if not self.notes:
            return "No notes saved."
        lines = []
        for i, n in enumerate(self.notes, 1):
            lines.append(f"{i}. {n['content']}")
        return "\n".join(lines)
    
    def _tool_save_last_image(self, filename: str = None) -> str:
        if self.pal.last_image is None:
            return "No image available to save."
        if filename is None:
            filename = f"capture_{int(time.time())}"
        path = SAVED_IMAGES_DIR / f"{filename}.jpg"
        cv2.imwrite(str(path), self.pal.last_image)
        return f"Image saved as {filename}.jpg"
    
    def _tool_set_timer(self, seconds: int, label: str = "timer") -> str:
        def fire():
            time.sleep(seconds)
            print(f"[Timer] {label} done!")
            # Could also queue a system message into the conversation
        threading.Thread(target=fire, daemon=True).start()
        return f"Timer set for {seconds} seconds: {label}"