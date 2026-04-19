from dataclasses import dataclass
from InputObject import InputObject

@dataclass
class Message:
    mid: int
    multimodal: bool
    messages: list[InputObject]