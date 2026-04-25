import threading
import cv2
import ollama
from Perception.voice.speak import voice_output

# Helper functions
def load_system_prompt(path):
    with open(path, "r") as f:
        system_prompt = f.read() 
    return system_prompt

def query_model(query, system_prompt=None, MODEL=None):
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

# Main interaction function
def run(query, system_prompt=None, model=None, muted=False):
    buffer = ""
    for token in query_model(query, system_prompt, model):
        #(token, end="", flush=True)
        buffer += token
        if buffer.endswith((".", "!", "?", "\n")):
            sentence = buffer.strip()
            print('[Debug] output chunk yielded')
            yield sentence
            print('[Debug] voice output now')
            voice_output(sentence)
            buffer = ""
    
    if buffer.strip():
        voice_output(buffer.strip())
        yield buffer.strip()

