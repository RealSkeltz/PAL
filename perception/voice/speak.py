import sounddevice as sd
from kokoro_onnx import Kokoro
kokoro = Kokoro("/Users/jscheltema/Documents/Personal/PAL/Shared/Resources/voice/kokoro-v0_19.onnx", "/Users/jscheltema/Documents/Personal/PAL/Shared/Resources/voice/voices.bin")

def speak(text, voice='af_bella'):
    samples, sample_rate = kokoro.create(text, voice=voice, speed=1.2)
    sd.play(samples, sample_rate)
    sd.wait()