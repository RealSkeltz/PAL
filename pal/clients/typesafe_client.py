from typesafe_sdk import Choice, Noul, Score, TypeSafeClient

class SystemOneClient():
    def __init__(self, model: str, think: bool = False):
        self.model = model

    def decide(self, state, questions):
        with TypeSafeClient(
            base_url="http://localhost:11434",
            api_key="ollama",          # ignored by ollama, but the SDK requires one
            timeout=120,
        ) as client:
            result = client.system_one(model=self.model, state=state, questions=questions)

        return result