import threading
import cv2
import ollama
from Perception.voice.speak import voice_output


# Helper functions
def load_system_prompt(path):
    with open(path, "r") as f:
        system_prompt = f.read()
    return system_prompt


def query_model(messages, system_prompt=None, MODEL=None):
    full_messages = [{"role": "system", "content": system_prompt}] + messages
    stream = ollama.chat(
        model=MODEL,
        messages=full_messages,
        think=False,
        stream=True,
    )
    for chunk in stream:
        yield chunk["message"]["content"]


# Main interaction function
def run(conversation, system_prompt=None, model=None, muted=False):
    buffer = ""
    for token in query_model(conversation, system_prompt, model):
        buffer += token
        if buffer.endswith((".", "!", "?", "\n")):
            sentence = buffer.strip()
            yield sentence
            voice_output(sentence)
            buffer = ""
    if buffer.strip():
        voice_output(buffer.strip())
        yield buffer.strip()