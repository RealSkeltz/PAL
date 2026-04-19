from Shared.Classes.Pal import Pal
import ollama

with open("/Users/jscheltema/Documents/Personal/PAL/Shared/Prompts/scout_identity.txt") as f:
    scout_identity = f.read()

class Scout(Pal):
    def __init__(self):
        super().__init__(system_prompt=scout_identity, model="qwen3.5:9b")