"""Shared utils file for helpful functions and objects for voice analysis
"""


import sounddevice as sd
import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from pal.config import SPEAKER_DATA


def record_sound(filename: str, duration: int):

    sr = 44100         # sample rate (Hz)
    channels = 1       # mono is best for voice analysis

    print("Recording...")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=channels, dtype="float32")
    sd.wait()
    print("Recording done.")

    sf.write(SPEAKER_DATA / "mine" / f"{filename}.wav", audio, sr, subtype="PCM_16")


def plot_spectogram(sound):
    # Plot spectrogram
    spec = sound.to_spectrogram()
    X, Y = spec.x_grid(), spec.y_grid()
    sg_db = 10 * np.log10(spec.values)
    plt.pcolormesh(X, Y, sg_db, vmin=sg_db.max() - 70, cmap="afmhot")
    plt.ylabel("Frequency (Hz)"); plt.xlabel("Time (s)")
    plt.show()

import torch

def transform_data(samples, mean, std):

    KEYS = list(samples[0].features_1.keys())

    def to_tensor(dicts):
        return torch.tensor([[float(d[k]) for k in KEYS] for d in dicts], dtype=torch.float32)
    
    # Prepare test data exactly like the training data
    T1 = (to_tensor(s.features_1 for s in samples) - mean) / std
    T2 = (to_tensor(s.features_2 for s in samples) - mean) / std
    X = torch.cat([(T1 - T2).abs(), T1 * T2], dim=1)
    y = torch.tensor([float(s.same) for s in samples])

    return X, y