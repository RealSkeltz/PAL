"""Shared voice-pair dataset code: pair sampling, acoustic features and train/test tensors
"""


import random
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import parselmouth
from parselmouth.praat import call


@dataclass
class DataSample:
    sample_1: Any
    sample_2: Any
    same: bool
    features_1: dict | None = None
    features_2: dict | None = None


def sample_pair(files_by_speaker: dict[str, list[Path]]) -> DataSample:
    """Random pair of clips, half the time from the same speaker"""
    speakers = list(files_by_speaker)
    same = random.random() < 0.5  # balance same / different speaker pairs

    speaker_1 = random.choice(speakers)
    if same:
        # two different clips from the same speaker
        sample_1, sample_2 = random.sample(files_by_speaker[speaker_1], 2)
    else:
        speaker_2 = random.choice([s for s in speakers if s != speaker_1])
        sample_1 = random.choice(files_by_speaker[speaker_1])
        sample_2 = random.choice(files_by_speaker[speaker_2])

    return DataSample(sample_1, sample_2, same)


@lru_cache(maxsize=None)  # each wav is analysed once, even if it shows up in many pairs
def extract_voice_features(file) -> dict:
    snd = parselmouth.Sound(str(file))
    features = {}

    # Pitch
    pitch = snd.to_pitch(pitch_floor=75, pitch_ceiling=500)
    f0 = pitch.selected_array["frequency"]
    f0_filt = f0[f0 > 0]

    features['f0_mean'] = np.mean(f0_filt)
    features['f0_std'] = np.std(f0_filt)
    features['f0_p5'], features['f0_p95'] = np.percentile(f0_filt, [5, 95])

    # Formants (median over voiced frames)
    formant = snd.to_formant_burg(maximum_formant=5500)
    times = pitch.xs()[f0 > 0]

    f = {n: np.nanmedian([formant.get_value_at_time(n, t) for t in times]) for n in range(1, 5)}
    for n, value in f.items():
        features[f'f{n}'] = value

    # Formant dispersion
    features['dispersion'] = (f[4] - f[1]) / 3

    # HNR
    harmonicity = snd.to_harmonicity_cc()
    features['hnr'] = call(harmonicity, "Get mean", 0, 0)

    # Jitter and shimmer
    pp = call(snd, "To PointProcess (periodic, cc)", 75, 500)
    features['jitter'] = call(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    features['shimmer'] = call([snd, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)

    # Spectral center of gravity
    spectrum = snd.to_spectrum()
    features['cog'] = call(spectrum, "Get centre of gravity", 2)

    return features

import torch
from torch.utils.data import TensorDataset, DataLoader, random_split

def create_train_data(samples):

    KEYS = list(samples[0].features_1.keys())

    def to_tensor(dicts):
        return torch.tensor([[float(d[k]) for k in KEYS] for d in dicts], dtype=torch.float32)

    X1 = to_tensor(s.features_1 for s in samples)
    X2 = to_tensor(s.features_2 for s in samples)
    y = torch.tensor([float(s.same) for s in samples])

    # Standardize: f4 is ~4000 while jitter is ~0.02, so scales must be equalized
    both = torch.cat([X1, X2])
    mean, std = both.mean(0), both.std(0) + 1e-8
    X1, X2 = (X1 - mean) / std, (X2 - mean) / std

    # Symmetric pair features: order of the two samples doesn't matter
    X = torch.cat([(X1 - X2).abs(), X1 * X2], dim=1)

    dataset = TensorDataset(X, y)
    n_val = int(0.2 * len(dataset))
    train_ds, val_ds = random_split(dataset, [len(dataset) - n_val, n_val])

    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=256)

    return train_dl, val_dl, mean, std