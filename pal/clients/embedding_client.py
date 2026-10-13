"""Embedding models"""

import ollama
from typing import Any

class EmbeddingClient():
    def __init__(self, model: str):
        self.model = model

    def vectorise(self, input: Any):
        
        response = ollama.embed(
            model=self.model,
            input=input,
        )

        return response


