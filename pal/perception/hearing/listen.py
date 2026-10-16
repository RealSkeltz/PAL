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

from pal.perception.hearing.vad import contains_speech

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
# Recognising the user takes some speech: from 0.5 s it works most of the time, and
# it is re-checked as more arrives. Other voices practically never pass. A short burst
# ("stop") is checked the moment it ends, since it never reaches 0.5 s.
CHECK_USER_FROM = round(0.5 / CHUNK_SEC)
CHECK_USER_EVERY = round(0.25 / CHUNK_SEC)

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

def check_user(speech_chunks, on_user_speaking) -> bool:
    """Early speaker check on the speech so far; signals once it recognises the user."""
    is_user, score = verify(voiceprint, np.concatenate(speech_chunks).reshape(-1), encoder, SPEAKER_THRESHOLD)
    logger.debug(f"early speaker check {score:.2f} after {len(speech_chunks) * CHUNK_SEC:.2f}s")
    if is_user:
        on_user_speaking()
    return is_user

def transcribe_audio(on_user_speaking):
    global chunk_id
    buffer = []
    speech = []          # only the chunks with speech, for the early speaker check
    user_confirmed = False
    silence_chunks = 0
    speech_chunks = 0
    in_speech = False
    pending = None 

    while True:
        chunk = audio_queue.get()

        if contains_speech(chunk):
            pending = None 
            buffer.append(chunk)
            speech.append(chunk)
            speech_chunks += 1
            if not user_confirmed and speech_chunks >= CHECK_USER_FROM \
                    and (speech_chunks - CHECK_USER_FROM) % CHECK_USER_EVERY == 0:
                user_confirmed = check_user(speech, on_user_speaking)
            silence_chunks = 0
            in_speech = True
        elif in_speech:
            
            buffer.append(chunk)
            silence_chunks += 1
            if silence_chunks == 1 and not user_confirmed and speech_chunks >= MIN_SPEECH_CHUNKS:
                user_confirmed = check_user(speech, on_user_speaking)  # a short burst just ended
        else:
            continue 

        # Speech so far, without the trailing silence
        speech_audio = lambda: np.concatenate(buffer[:len(buffer) - silence_chunks]).reshape(-1)

        if silence_chunks == SPECULATE_AFTER and speech_chunks >= MIN_SPEECH_CHUNKS:
            pending = worker.submit(process, speech_audio())

        if silence_chunks >= COMMIT_AFTER or len(buffer) >= MAX_BUFFER_CHUNKS:
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
            buffer = []
            speech = []
            user_confirmed = False
            silence_chunks = 0
            speech_chunks = 0
            in_speech = False
            pending = None

def run(on_user_speaking=lambda: None) -> Generator[Spoken, None, None]:
    """Yield one Spoken event per utterance. `on_user_speaking` fires once per utterance,
    as soon as enough speech has come in to recognise it as the user's voice."""
    threading.Thread(target=transcribe_audio, args=(on_user_speaking,), daemon=True).start()
    with sd.InputStream(samplerate=16000, channels=1, dtype='float32',
                        blocksize=512, callback=callback):
        while True:
            yield result_queue.get()
        