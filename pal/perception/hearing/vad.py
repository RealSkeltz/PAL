"""Voice activity detection: is there speech in this bit of audio? (Silero VAD)

Used live by the listener, chunk by chunk, and offline to cut the silences out
of a recording before making a voiceprint.
"""
import numpy as np
import torch
from silero_vad import load_silero_vad

SAMPLE_RATE = 16000
CHUNK = 512          # samples per decision, 32 ms at 16 kHz
SPEECH_PROB = 0.5

_model = load_silero_vad()


@torch.inference_mode()
def contains_speech(chunk, sr=SAMPLE_RATE) -> bool:
    """chunk: 512 samples of 16 kHz mono float32 (shape (512,) or (512, 1))"""
    wave = torch.from_numpy(np.asarray(chunk, dtype=np.float32).reshape(-1))
    return _model(wave, sr).item() >= SPEECH_PROB


def speech_only(audio) -> np.ndarray:
    """A 16 kHz recording with the silences cut out, the way the listener buffers speech.

    Resets the detector's running state, so don't call it while the listener is live.
    """
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    _model.reset_states()
    kept = [c for c in (audio[i:i + CHUNK] for i in range(0, len(audio) - CHUNK + 1, CHUNK)) if contains_speech(c)]
    _model.reset_states()
    return np.concatenate(kept) if kept else np.zeros(0, dtype=np.float32)
