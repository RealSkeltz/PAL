"""Synthetic voice data: Kokoro TTS clips per voice and synthetic voice-pair samples
"""


from pathlib import Path

import soundfile as sf

from data_generation import DataSample, sample_pair


TTS_DIR = Path(__file__).parent / "data" / "tts"
SENTENCES_FILE = Path(__file__).parent / "resources" / "sentences.txt"

# One sentence per line, id = line number (matches the 001.wav ... file names)
SENTENCES = {i: line for i, line in enumerate(SENTENCES_FILE.read_text().splitlines(), 1) if line.strip()}


def generate_tts(voices: list[str] | None = None, speed: float = 1.0) -> dict:
    """Say every sentence with every voice and save as data/tts/<voice>/<id>.wav

    Already generated files are skipped, so re-running resumes.
    Returns tts[voice][sentence_id] = Path to the wav file.
    """
    # Imported here so the rest of this module works without loading the TTS model
    from pal.voice.speak import kokoro

    voices = voices or list(kokoro.voices)
    tts = {}

    for vi, voice in enumerate(voices, 1):
        voice_dir = TTS_DIR / voice
        voice_dir.mkdir(parents=True, exist_ok=True)
        tts[voice] = {}

        for i, text in SENTENCES.items():
            path = voice_dir / f"{i:03d}.wav"
            if not path.exists():
                samples, sample_rate = kokoro.create(text, voice=voice, speed=speed)
                sf.write(path, samples, sample_rate, subtype="PCM_16")
            tts[voice][i] = path

        print(f"[{vi}/{len(voices)}] {voice} done")

    return tts


def voice_files(voice: str) -> list[Path]:
    return sorted((TTS_DIR / voice).glob("*.wav"))


def available_voices() -> list[str]:
    """Voices on disk with at least two clips, so a same-voice pair can be drawn"""
    return sorted(d.name for d in TTS_DIR.iterdir() if d.is_dir() and len(voice_files(d.name)) >= 2)


def synthetic_files_by_voice() -> dict[str, list[Path]]:
    return {voice: voice_files(voice) for voice in available_voices()}


def generate_synthetic_sample(files_by_voice: dict[str, list[Path]] | None = None) -> DataSample:
    return sample_pair(files_by_voice or synthetic_files_by_voice())
