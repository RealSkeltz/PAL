from dataclasses import dataclass
from Shared.Classes.InputObject import InputObject

@dataclass
class Message:
    mid: int
    multimodal: bool
    messages: list[InputObject]