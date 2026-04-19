from abc import ABC, abstractmethod
from Shared.Classes.Message import Message
from Perception.video import vision_loop
from Perception.audio import audio_loop

class Pal(ABC):
    def __init__(self, system_prompt: str):
        # Identity
        self.system_prompt = system_prompt

        # Input processing
        self.internal_queue = []
        self.queue_history = []

    # Main loop
    def run(self):
        while True:
            if len(self.internal_queue) > 0:
                message = self.internal_queue[0]
                self.interact(message)
                self.queue_history.append(message)

    # Senses
    def see(self):
        vision_loop.run()

    def listen(self):
        audio_loop.run()

    @abstractmethod
    def speak(self):
        pass

    @abstractmethod
    def interact(self, message: Message):
        pass
