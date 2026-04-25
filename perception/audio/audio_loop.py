import sounddevice as sd
import numpy as np
import threading
import queue
from faster_whisper import WhisperModel
from typing import Generator
from Shared.Classes.Message import Message
from Shared.Classes.InputObject import InputObject


model = WhisperModel("tiny", device="cpu", compute_type="int8")

audio_queue = queue.Queue()
transcript = {}  # chunk_id -> text
chunk_id = 0

result_queue = queue.Queue()

def callback(indata, frames, time, status):
    audio_queue.put(indata.copy())

def transcribe_audio():
    global chunk_id
    buffer = []
    silence_count = 0
    
    while True:
        chunk = audio_queue.get()
        buffer.append(chunk)
        
        # Check if silent
        rms = np.sqrt(np.mean(chunk**2))
        if rms < 0.01:
            silence_count += 1
        else:
            silence_count = 0
        
        # Transcribe every ~2 seconds of speech, or on silence
        if len(buffer) > 15 or (silence_count > 15 and buffer):
            audio = np.concatenate(buffer).squeeze().astype(np.float32)
            buffer = []
            silence_count = 0
            
            if len(audio) < 4000:
                continue
                
            segments, info = model.transcribe(
                audio,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500, threshold=0.5),
                language='en'
            )
            text = " ".join(s.text for s in segments).strip()
            
            if text:
                transcript[chunk_id] = text
                chunk_id += 1
                # Print live
                #print("\r" + " ".join(transcript.values()), end="", flush=True)
                result_queue.put(text)

def run() -> Generator[InputObject, None, None]:
    threading.Thread(target=transcribe_audio, daemon=True).start()
    with sd.InputStream(samplerate=16000, channels=1, dtype='float32',
                        blocksize=1600, callback=callback):
        while True:
            text = result_queue.get()
            yield InputObject(type="audio", content=text)
        