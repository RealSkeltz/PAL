from abc import ABC, abstractmethod

class Pal(ABC):
    def __init__(self):
        self.internal_queue = []

    # Identity
    @property
    @abstractmethod
    def system_prompt(self):
        pass
    
    # Interaction
    @abstractmethod
    def listen(self):
        pass

    @abstractmethod
    def speak(self):
        pass

    @abstractmethod
    def interact(self):
        pass
