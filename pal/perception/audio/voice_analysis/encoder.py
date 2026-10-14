import math
import random

import parselmouth
import torch
from torch import nn
from torch.utils.data import Dataset

# _________ Audio preprocessing ________________________

SR = 16000       # LibriSpeech sample rate; other audio gets resampled to this
N_FFT = 512      # 32 ms analysis window
HOP = 160        # 10 ms between frames
N_MELS = 80
CROP_SEC = 3.0

def hz_to_mel(f):
    return 2595 * torch.log10(1 + f / 700)

def mel_to_hz(m):
    return 700 * (10 ** (m / 2595) - 1)

def mel_filterbank(n_mels=N_MELS, n_fft=N_FFT, sr=SR, f_min=20.0):
    fft_freqs = torch.linspace(0, sr / 2, n_fft // 2 + 1)
    mels = torch.linspace(hz_to_mel(torch.tensor(f_min)), hz_to_mel(torch.tensor(sr / 2)), n_mels + 2)
    hz = mel_to_hz(mels)
    lower, center, upper = hz[:-2, None], hz[1:-1, None], hz[2:, None]
    rising = (fft_freqs - lower) / (center - lower)
    falling = (upper - fft_freqs) / (upper - center)
    return torch.clamp(torch.minimum(rising, falling), min=0)  # (n_mels, n_fft // 2 + 1)

MEL_FB = mel_filterbank()
WINDOW = torch.hann_window(N_FFT)

def log_mel(wave):  # wave: 1D tensor
    spec = torch.stft(wave, N_FFT, hop_length=HOP, window=WINDOW, return_complex=True)
    mel = MEL_FB @ spec.abs() ** 2               # (n_mels, frames)
    log_mel = torch.log(mel + 1e-6)
    return log_mel - log_mel.mean(dim=1, keepdim=True)  # mean normalization per band

def load_wave(path):
    snd = parselmouth.Sound(str(path)).convert_to_mono()
    if snd.sampling_frequency != SR:
        snd = snd.resample(SR)  # e.g. TTS output is often 24 kHz
    return torch.tensor(snd.values[0], dtype=torch.float32)

def random_crop(wave, n_samples):
    if len(wave) < n_samples:  # repeat short clips instead of padding with silence
        wave = wave.repeat(math.ceil(n_samples / len(wave)))
    start = random.randint(0, len(wave) - n_samples)
    return wave[start:start + n_samples]

class SpeakerDataset(Dataset):
    def __init__(self, files_by_speaker, crop_sec=CROP_SEC):
        self.speakers = sorted(files_by_speaker)
        self.items = [(f, i) for i, s in enumerate(self.speakers) for f in files_by_speaker[s]]
        self.n_samples = int(crop_sec * SR)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        wave = random_crop(load_wave(path), self.n_samples)
        return log_mel(wave), label


# _________ Model ________________________

class SpeakerEncoder(nn.Module):
    def __init__(self, n_mels=80, emb_dim=192, n_speakers=251):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_mels, 256, 5, padding=2), nn.ReLU(), nn.BatchNorm1d(256),
            nn.Conv1d(256, 256, 3, padding=2, dilation=2), nn.ReLU(), nn.BatchNorm1d(256),
            nn.Conv1d(256, 256, 3, padding=3, dilation=3), nn.ReLU(), nn.BatchNorm1d(256),
        )
        self.embed = nn.Linear(512, emb_dim)
        self.classify = nn.Linear(emb_dim, n_speakers)  # only used during training

    def embedding(self, mel):  # mel: (batch, n_mels, frames)
        h = self.conv(mel)
        stats = torch.cat([h.mean(dim=2), h.std(dim=2)], dim=1)  # statistics pooling
        return self.embed(stats)

    def forward(self, mel):
        return self.classify(self.embedding(mel))
