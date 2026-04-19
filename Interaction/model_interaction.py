from Perception.audio.audio_loop import listen
from Perception.video.listen import listen

import ollama

# Helper functions
def load_system_prompt(path):
    with open(path, "r") as f:
        system_prompt = f.read() 
    return system_prompt

def query_model(query, system_prompt=None, MODEL=None):
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

# Main interaction function
def interact(query, system_prompt=None, MODEL=None):
    buffer = ""
    for token in query_model(query, system_prompt, MODEL):
        print(token, end="", flush=True)
        buffer += token
        if buffer.endswith((".", "!", "?", "\n")):
            speak(buffer.strip())
            buffer = ""
    if buffer.strip():
        speak(buffer.strip())

