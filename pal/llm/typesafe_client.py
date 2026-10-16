"""System One models (choice/score questions) served by ollama.

Like the ollama client, this keeps its model loaded while PAL runs: a cold
load takes seconds, a warm decision takes milliseconds.
"""
import ollama
from typesafe_sdk import TypeSafeClient

OLLAMA_URL = "http://localhost:11434"
WARM_UP_QUESTION = {"warm_up": {"type": "choice", "instructions": "Is this a test?",
                                "criteria": {"yes": "It is a test.", "no": "It is not a test."}}}


class SystemOneClient():
    def __init__(self, model: str, think: bool = False):
        self.model = model

    def decide(self, state, questions):
        with TypeSafeClient(
            base_url=OLLAMA_URL,
            api_key="ollama",          # ignored by ollama, but the SDK requires one
            timeout=120,
        ) as client:
            return client.system_one(
                model=self.model, state=state, questions=questions,
                extra_body={"keep_alive": -1},
            )

    def warm_up(self) -> bool:
        # System One models refuse ollama's plain load request, so make one throwaway decision.
        self.decide("warm up", WARM_UP_QUESTION)
        return True

    def release(self):
        ollama.generate(model=self.model, keep_alive=0)
