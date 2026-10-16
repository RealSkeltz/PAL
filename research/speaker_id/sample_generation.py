"""Voice-pair samples with features, drawn from synthetic TTS clips or LibriSpeech
"""


import json
from functools import lru_cache
from pathlib import Path

from pal.config import SPEAKER_DATA
from data_generation import DataSample, extract_voice_features, sample_pair
from synthetic_data_generation import synthetic_files_by_voice


BASE_DIR = SPEAKER_DATA  # sample files store paths relative to this
LIBRISPEECH_DIR = BASE_DIR / "slr" / "LibriSpeech"
SAMPLES_DIR = BASE_DIR / "samples"


@lru_cache(maxsize=None)
def librispeech_files_by_speaker(subset: str = "dev-clean") -> dict[str, list[Path]]:
    """speaker id -> all of that speaker's clips (chapters are flattened, they don't matter here)"""
    files_by_speaker = {}

    for speaker_dir in sorted((LIBRISPEECH_DIR / subset).iterdir()):
        if speaker_dir.is_dir():
            files = sorted(speaker_dir.glob("*/*.flac"))  # <speaker>/<chapter>/<file>.flac
            if len(files) >= 2:
                files_by_speaker[speaker_dir.name] = files

    return files_by_speaker


def generate_librispeech_sample(files_by_speaker: dict[str, list[Path]] | None = None) -> DataSample:
    return sample_pair(files_by_speaker or librispeech_files_by_speaker())


SOURCES = {
    "synthetic": synthetic_files_by_voice,
    "librispeech": librispeech_files_by_speaker,
}


def generate_samples(num: int, source: str = "synthetic") -> list[DataSample]:
    """num voice pairs with features, from source "synthetic" or "librispeech" """
    files_by_speaker = SOURCES[source]()
    samples = []

    for _ in range(num):
        sample = sample_pair(files_by_speaker)
        sample.features_1 = extract_voice_features(sample.sample_1)
        sample.features_2 = extract_voice_features(sample.sample_2)
        samples.append(sample)

    return samples


def save_samples(samples: list[DataSample], path: Path):
    """Store samples as JSON; clip paths are relative to this folder, features as plain floats"""
    rows = [
        {
            "sample_1": str(Path(s.sample_1).relative_to(BASE_DIR)),
            "sample_2": str(Path(s.sample_2).relative_to(BASE_DIR)),
            "same": s.same,
            "features_1": {k: float(v) for k, v in s.features_1.items()},
            "features_2": {k: float(v) for k, v in s.features_2.items()},
        }
        for s in samples
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows))


def load_samples(path: Path) -> list[DataSample]:
    rows = json.loads(path.read_text())
    return [
        DataSample(BASE_DIR / r["sample_1"], BASE_DIR / r["sample_2"], r["same"], r["features_1"], r["features_2"])
        for r in rows
    ]
