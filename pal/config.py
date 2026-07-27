"""Filesystem locations, resolved relative to the repo root.

Everything that reads or writes a file goes through here so nothing has to
hardcode an absolute path.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

MODELS_DIR = ROOT / "models"
VISION_MODELS = MODELS_DIR / "vision"
VOICE_MODELS = MODELS_DIR / "voice"

DATA_DIR = ROOT / "data"
NOTES_FILE = DATA_DIR / "notes.json"
SAVED_IMAGES_DIR = DATA_DIR / "saved_images"

DATA_DIR.mkdir(parents=True, exist_ok=True)
SAVED_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
