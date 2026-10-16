"""Kokoro text-to-speech, loaded once and shared."""
from kokoro_onnx import Kokoro

from pal.config import VOICE_MODELS

kokoro = Kokoro(
    str(VOICE_MODELS / "kokoro-v0_19.onnx"),
    str(VOICE_MODELS / "voices.bin"),
)
