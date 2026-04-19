from dataclasses import dataclass
from typing import Any

@dataclass
class InputObject:
    oid: int
    type: str
    content: Any
