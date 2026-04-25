from Shared.Classes.Pal import Pal
import ollama

from Shared.Context.context_loader import build_system_prompt

system_prompt = build_system_prompt()

class Scout(Pal):
    def __init__(self):
        super().__init__(system_prompt=system_prompt, model="qwen3.5:9b")