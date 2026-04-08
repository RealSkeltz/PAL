import platform
import subprocess

# Language Model
import ollama
MODEL = "qwen3.5:9b"

# Voice model
from kokoro_onnx import Kokoro
import sounddevice as sd
kokoro = Kokoro("/Users/jscheltema/Documents/Personal/PAL/agent/voice/kokoro-v0_19.onnx", "/Users/jscheltema/Documents/Personal/PAL/agent/voice/voices.bin")


def load_system_prompt():
    with open("prompts/system_prompt.txt", "r") as f:
        system_prompt = f.read() 
    return system_prompt

def query_model(query, system_prompt=None):
    system_prompt = load_system_prompt()
    
    stream = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ],
        think=False,
        stream=True,
    )
    for chunk in stream:
        yield chunk["message"]["content"]

def speak_mac(text):
    if platform.system() == "Darwin":
        subprocess.run(["say", "-v", "Daniel", text])
    else:
        subprocess.run(["espeak", text])

def speak(text, voice):
    samples, sample_rate = kokoro.create(text, voice=voice, speed=1.2)
    sd.play(samples, sample_rate)
    sd.wait()

def interact(query, voice):
    buffer = ""
    for token in query_model(query):
        print(token, end="", flush=True)
        buffer += token
        if buffer.endswith((".", "!", "?", "\n")):
            speak(buffer.strip(), voice)
            buffer = ""
    if buffer.strip():
        speak(buffer.strip())