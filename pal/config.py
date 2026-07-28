"""Filesystem locations, resolved relative to the repo root.

Everything that reads or writes a file goes through here so nothing has to
hardcode an absolute path.
"""
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent

# Credentials and model selection live in .env (gitignored). Loaded here because
# config is imported early — before any LLM client reads the environment.
# Real environment variables take precedence over the file.
load_dotenv(ROOT / ".env")

MODELS_DIR = ROOT / "models"
VISION_MODELS = MODELS_DIR / "vision"
VOICE_MODELS = MODELS_DIR / "voice"

DATA_DIR = ROOT / "data"
NOTES_FILE = DATA_DIR / "notes.json"
SAVED_IMAGES_DIR = DATA_DIR / "saved_images"

DATA_DIR.mkdir(parents=True, exist_ok=True)
SAVED_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
