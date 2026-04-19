import sounddevice as sd

#_______ In __________
import numpy as np
import threading
import queue
from faster_whisper import WhisperModel

model = WhisperModel("tiny", device="cpu", compute_type="int8")

audio_queue = queue.Queue()
transcript = {}  # chunk_id -> text
chunk_id = 0

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
                
            segments, _ = model.transcribe(audio, language="en")
            text = " ".join(s.text for s in segments).strip()
            
            if text:
                transcript[chunk_id] = text
                chunk_id += 1
                # Print live
                print("\r" + " ".join(transcript.values()), end="", flush=True)

def listen():
    # Run transcription in background thread
    t = threading.Thread(target=transcribe_audio, daemon=True)
    t.start()

    print("Speak now... (Ctrl+C to stop)")
    with sd.InputStream(samplerate=16000, channels=1, dtype='float32', 
                        blocksize=1600, callback=callback):
        try:
            while True:
                pass
        except KeyboardInterrupt:
            print("\nStopped.")

#_______ Out __________
from kokoro_onnx import Kokoro
kokoro = Kokoro("/Users/jscheltema/Documents/Personal/PAL/agent/voice/kokoro-v0_19.onnx", "/Users/jscheltema/Documents/Personal/PAL/agent/voice/voices.bin")

def speak(text, voice='af_bella'):
    samples, sample_rate = kokoro.create(text, voice=voice, speed=1.5)
    sd.play(samples, sample_rate)
    sd.wait()