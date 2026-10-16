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
SPEAKER_MODELS = MODELS_DIR / "speaker"
VOICEPRINTS_DIR = SPEAKER_MODELS / "voiceprints"

DATA_DIR = ROOT / "data"
NOTES_FILE = DATA_DIR / "notes.json"
SAVED_IMAGES_DIR = DATA_DIR / "saved_images"
CAPTURES_DIR = DATA_DIR / "captures"
LOGS_DIR = DATA_DIR / "logs"
SPEAKER_DATA = DATA_DIR / "speaker_id"  # training data for the speaker encoder

DATA_DIR.mkdir(parents=True, exist_ok=True)
SAVED_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
CAPTURES_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
