from Interaction.in_out import listen, speak

# Language Model
import ollama
MODEL = "qwen3.5:9b"

# Helper functions
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

# Main interaction function
def interact(query):
    buffer = ""
    for token in query_model(query):
        print(token, end="", flush=True)
        buffer += token
        if buffer.endswith((".", "!", "?", "\n")):
            speak(buffer.strip())
            buffer = ""
    if buffer.strip():
        speak(buffer.strip())

