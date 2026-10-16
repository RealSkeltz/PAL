import numpy as np
import sounddevice as sd
from kokoro_onnx import Kokoro
from pal.config import VOICE_MODELS
from pal.utils.timing import timed

kokoro = Kokoro(
    str(VOICE_MODELS / "kokoro-v0_19.onnx"),
    str(VOICE_MODELS / "voices.bin"),
)

# Open once, reuse for every chunk. Kokoro outputs 24kHz mono float32.
_output_stream = sd.OutputStream(
    samplerate=24000,
    channels=1,
    dtype='float32',
    blocksize=1024,
)
_output_stream.start()


def voice_output(text, voice='af_bella'):
    with timed(f"tts ({len(text)} chars)"):
        samples, sample_rate = kokoro.create(text, voice=voice, speed=1.2)
    # samples is float32 mono; ensure shape is (N,) or (N, 1)
    _output_stream.write(samples)

# --- Acknowledgment tones ---

def play_sound(samples):
    _output_stream.write(samples)