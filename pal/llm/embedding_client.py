"""Text embeddings from a local ollama model, used for recall.

Ported from the memory branch. Vectors come back L2-normalised, so the dot
product of two of them is their cosine similarity.
"""
import numpy as np
import ollama


class EmbeddingClient:
    def __init__(self, model: str):
        self.model = model

    def vectorise(self, texts: list[str]) -> np.ndarray:
        """(len(texts), dim) float32, one normalised row per text."""
        response = ollama.embed(model=self.model, input=texts, keep_alive=-1)
        vectors = np.asarray(response["embeddings"], dtype=np.float32)
        return vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-9)

    def warm_up(self) -> bool:
        self.vectorise(["warm up"])
        return True

    def release(self):
        ollama.embed(model=self.model, input=["release"], keep_alive=0)
