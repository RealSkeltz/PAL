import logging
import sounddevice as sd
import numpy as np
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
from faster_whisper import WhisperModel
from typing import Generator
from pal.config import VOICEPRINTS_DIR
from pal.schemas.event import Actor, Role, Spoken
from pal.perception.hearing.whisper_prompt import WHISPER_PROMPT

from pal.debug.timing import timed

logger = logging.getLogger(__name__)

from pal.perception.hearing.speaker_encoder import verify, load_encoder
import torch

voiceprint = torch.load(VOICEPRINTS_DIR / "voiceprint_me.pt")
encoder, _ = load_encoder()

SPEAKER_THRESHOLD = 0.5

transcriber = WhisperModel("distil-small.en", device="auto", compute_type="float32", cpu_threads=4)

from silero_vad import load_silero_vad
vad = load_silero_vad()
SPEECH_PROB = 0.5

@torch.inference_mode()
def contains_speech(chunk, sr=16000) -> bool:
    """chunk: 512 samples of 16 kHz mono float32 (shape (512,) or (512, 1))"""
    wave = torch.from_numpy(chunk.reshape(-1))
    return vad(wave, sr).item() >= SPEECH_PROB

audio_queue = queue.Queue()
transcript = {}  # chunk_id -> text
chunk_id = 0

result_queue = queue.Queue()

def callback(indata, frames, time, status):
    audio_queue.put(indata.copy())

CHUNK_SEC = 512 / 16000
SPECULATE_AFTER = round(0.2 / CHUNK_SEC)    # short pause: start working on what was said so far
COMMIT_AFTER = round(0.5 / CHUNK_SEC)       # long pause: the utterance is finished
MIN_SPEECH_CHUNKS = round(0.3 / CHUNK_SEC)  # less speech than this is a blip, not an utterance
MAX_BUFFER_CHUNKS = round(20 / CHUNK_SEC)   # safety cap for one utterance

worker = ThreadPoolExecutor(max_workers=1)  # one Whisper run at a time

def process(audio):
    is_user, score = verify(voiceprint, audio, encoder, SPEAKER_THRESHOLD)
    logger.info(f"speaker score {score:.2f} ({len(audio) / 16000:.1f}s) -> {'user' if is_user else 'other'}")

    with timed("STT"):
        segments, info = transcriber.transcribe(
            audio,
            language='en',
            condition_on_previous_text=True,
            initial_prompt=WHISPER_PROMPT,
        )
        text = " ".join(s.text for s in segments).strip()
    return text, is_user, float(score), len(audio) / 16000

def transcribe_audio(on_speech_start, on_speech_dropped):
    global chunk_id
    buffer = []
    silence_chunks = 0
    speech_chunks = 0
    in_speech = False
    pending = None 

    while True:
        chunk = audio_queue.get()

        if contains_speech(chunk):
            if not in_speech:
                on_speech_start()
            pending = None 
            buffer.append(chunk)
            speech_chunks += 1
            silence_chunks = 0
            in_speech = True
        elif in_speech:
            
            buffer.append(chunk)
            silence_chunks += 1
        else:
            continue 

        # Speech so far, without the trailing silence
        speech_audio = lambda: np.concatenate(buffer[:len(buffer) - silence_chunks]).reshape(-1)

        if silence_chunks == SPECULATE_AFTER and speech_chunks >= MIN_SPEECH_CHUNKS:
            pending = worker.submit(process, speech_audio())

        if silence_chunks >= COMMIT_AFTER or len(buffer) >= MAX_BUFFER_CHUNKS:
            text = None
            if speech_chunks >= MIN_SPEECH_CHUNKS:
                text, is_user, score, duration = pending.result() if pending is not None else process(speech_audio())
                if text:
                    transcript[chunk_id] = text
                    chunk_id += 1
                    result_queue.put(Spoken(
                        actor=Actor.USER if is_user else Actor.OTHER,
                        role=Role.CONTEXT,  # the router's gate decides what is direct
                        text=text,
                        duration_s=duration,
                        speaker_score=score,
                    ))
            if not text:
                on_speech_dropped()  # a blip or no words: nothing will reach the router
            buffer = []
            silence_chunks = 0
            speech_chunks = 0
            in_speech = False
            pending = None

def run(on_speech_start=lambda: None, on_speech_dropped=lambda: None) -> Generator[Spoken, None, None]:
    """Yield one Spoken event per utterance. `on_speech_start` fires the moment speech is
    detected, `on_speech_dropped` when it turns out to be too short or wordless."""
    threading.Thread(target=transcribe_audio, args=(on_speech_start, on_speech_dropped), daemon=True).start()
    with sd.InputStream(samplerate=16000, channels=1, dtype='float32',
                        blocksize=512, callback=callback):
        while True:
            yield result_queue.get()
        