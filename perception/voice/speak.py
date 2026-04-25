import sounddevice as sd
from kokoro_onnx import Kokoro
kokoro = Kokoro(
    "/Users/jscheltema/Documents/Personal/PAL/Shared/Resources/voice_models/kokoro-v0_19.onnx",
    "/Users/jscheltema/Documents/Personal/PAL/Shared/Resources/voice_models/voices.bin"
)
def voice_output(text, voice='af_bella'):
    samples, sample_rate = kokoro.create(text, voice=voice, speed=0.5)
    sd.play(samples, sample_rate)
    sd.wait()