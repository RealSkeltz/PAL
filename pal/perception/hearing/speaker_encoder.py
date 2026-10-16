import math
import random
from pathlib import Path

import numpy as np
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

def to_wave(audio, sr=SR):
    """In-memory audio (numpy array or tensor, e.g. from sounddevice) -> 1D float32 tensor at SR"""
    audio = np.asarray(audio, dtype=np.float64)
    if audio.ndim == 2:  # (samples, channels) as recorded by sounddevice
        audio = audio.mean(axis=1)
    if sr != SR:
        audio = parselmouth.Sound(audio, sampling_frequency=sr).resample(SR).values[0]
    return torch.tensor(audio, dtype=torch.float32)

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

    

class SmallSpeakerEncoder(nn.Module):
    def __init__(self, n_mels=80, channels=64, emb_dim=64, n_speakers=251):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_mels, channels, 5, padding=2), nn.ReLU(), nn.BatchNorm1d(channels),
            nn.Conv1d(channels, channels, 3, padding=2, dilation=2), nn.ReLU(), nn.BatchNorm1d(channels),
        )
        self.embed = nn.Linear(2 * channels, emb_dim)
        self.classify = nn.Linear(emb_dim, n_speakers)


    def embedding(self, mel):  # mel: (batch, n_mels, frames)
        h = self.conv(mel)
        stats = torch.cat([h.mean(dim=2), h.std(dim=2)], dim=1)  # statistics pooling
        return self.embed(stats)

    def forward(self, mel):
        return self.classify(self.embedding(mel))

    

# ______ Serving the model ________________________

import torch.nn.functional as F

@torch.inference_mode()
def embed_wave(wave, model):
    mel = log_mel(wave).unsqueeze(0)
    return F.normalize(model.embedding(mel), dim=-1).squeeze(0)

def embed_file(path, model):
    return embed_wave(load_wave(path), model)

def embed_sound(audio, model, sr=SR):
    """Audio already in memory, e.g. a microphone recording (live input is 16 kHz mono float32)"""
    return embed_wave(to_wave(audio, sr), model)

def embed(audio, model, sr=SR):
    """File path or in-memory audio, so the functions below accept both"""
    if isinstance(audio, (str, Path)):
        return embed_file(audio, model)
    return embed_sound(audio, model, sr)


def same_speaker(audio_a, audio_b, model, threshold, sr=SR):
    score = torch.dot(embed(audio_a, model, sr), embed(audio_b, model, sr)).item()
    return score >= threshold, score

def enroll(audios, model, sr=SR):
    voiceprint = torch.stack([embed(a, model, sr) for a in audios]).mean(dim=0)
    return F.normalize(voiceprint, dim=0)

def verify(voiceprint, audio, model, threshold, sr=SR):
    score = torch.dot(voiceprint, embed(audio, model, sr)).item()
    return score >= threshold, score


# _________ Saving / loading ________________________

MODELS_DIR = Path(__file__).parents[1] / "data" / "models"
AUDIO_CONFIG = {"sr": SR, "n_fft": N_FFT, "hop": HOP, "n_mels": N_MELS}
ENCODERS = {cls.__name__: cls for cls in (SpeakerEncoder, SmallSpeakerEncoder)}


def save_encoder(model, threshold, path=MODELS_DIR / "speaker_encoder.pt"):
    """Store weights plus everything needed to rebuild and use the encoder"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    config = {
        "n_mels": model.conv[0].in_channels,
        "emb_dim": model.embed.out_features,
        "n_speakers": model.classify.out_features,
    }
    if isinstance(model, SmallSpeakerEncoder):
        config["channels"] = model.conv[0].out_channels

    torch.save({
        "architecture": type(model).__name__,
        "model_config": config,
        "model_state": model.state_dict(),
        "threshold": threshold,
        "audio_config": AUDIO_CONFIG,
    }, path)


def load_encoder(path=MODELS_DIR / "speaker_encoder.pt"):
    """Returns (model in eval mode, same-speaker threshold)"""
    ckpt = torch.load(path, map_location="cpu")
    assert ckpt["audio_config"] == AUDIO_CONFIG, "Audio settings differ from training"

    # Older checkpoints don't store the architecture; only the big encoder has a third conv block
    architecture = ckpt.get("architecture") or (
        "SpeakerEncoder" if "conv.6.weight" in ckpt["model_state"] else "SmallSpeakerEncoder"
    )
    model = ENCODERS[architecture](**ckpt["model_config"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    return model, ckpt["threshold"]
