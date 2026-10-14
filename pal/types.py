from dataclasses import dataclass


@dataclass
class InputObject:
    type: str
    content: any
    is_user: bool
    oid: int = 0


@dataclass
class Message:
    mid: int
    multimodal: bool
    messages: list[InputObject]
