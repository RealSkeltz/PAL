import sounddevice as sd
import numpy as np
import threading
import queue
from faster_whisper import WhisperModel
from typing import Generator
from pal.types import Message
from pal.types import InputObject
from pal.constants.whisper_prompt import WHISPER_PROMPT

from pal.utils.timing import timed

model = WhisperModel("distil-small.en", device="auto", compute_type="float32", cpu_threads=4)

audio_queue = queue.Queue()
transcript = {}  # chunk_id -> text
chunk_id = 0

result_queue = queue.Queue()

def callback(indata, frames, time, status):
    audio_queue.put(indata.copy())

def transcribe_audio():
    global chunk_id
    buffer = []
    silence_chunks = 0
    speech_chunks = 0
    
    # Tunables (chunks of 0.1s each at blocksize=1600, sr=16000)
    SILENCE_TO_FLUSH = 7      # 0.8s of silence ends an utterance
    MIN_SPEECH_CHUNKS = 3      # require ~0.3s of speech before considering flush
    MAX_BUFFER_CHUNKS = 200    # 20s safety cap to prevent runaway buffer
    RMS_THRESHOLD = 0.01
    
    in_speech = False
    
    while True:
        chunk = audio_queue.get()
        rms = np.sqrt(np.mean(chunk**2))
        is_speech = rms >= RMS_THRESHOLD
        
        if is_speech:
            buffer.append(chunk)
            speech_chunks += 1
            silence_chunks = 0
            in_speech = True
        elif in_speech:
            # In an utterance but currently silent — keep buffering trailing silence
            buffer.append(chunk)
            silence_chunks += 1
        # else: pre-speech silence, drop it
        
        # Flush when we've had enough silence after speech, or buffer is huge
        should_flush = (
            in_speech and silence_chunks >= SILENCE_TO_FLUSH and speech_chunks >= MIN_SPEECH_CHUNKS
        ) or len(buffer) >= MAX_BUFFER_CHUNKS
        
        if should_flush:
            audio = np.concatenate(buffer).squeeze().astype(np.float32)
            buffer = []
            silence_chunks = 0
            speech_chunks = 0
            in_speech = False
            
            if len(audio) < 4000:
                continue
            
            with timed("STT"):
                segments, info = model.transcribe(
                    audio,
                    vad_filter=False,
                    vad_parameters=dict(min_silence_duration_ms=500, threshold=0.5),
                    language='en',
                    condition_on_previous_text=False,
                    initial_prompt=WHISPER_PROMPT,
                )
                text = " ".join(s.text for s in segments).strip()
            if text:
                transcript[chunk_id] = text
                chunk_id += 1
                result_queue.put(text)

def run() -> Generator[InputObject, None, None]:
    threading.Thread(target=transcribe_audio, daemon=True).start()
    with sd.InputStream(samplerate=16000, channels=1, dtype='float32',
                        blocksize=1600, callback=callback):
        while True:
            text = result_queue.get()
            yield InputObject(type="audio", content=text)
        