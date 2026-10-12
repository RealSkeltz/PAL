"""Local models served by ollama."""

from typesafe_sdk import Choice, Noul, Score, TypeSafeClient

class SystemOneClient():
    def __init__(self, model: str):
        self.model = model

    def decide(self, state: str, questions: dict):
        with TypeSafeClient(timeout=120) as client:
            result = client.system_one(
                model=self.model,
                state=state,
                questions=questions
            )

        return result


