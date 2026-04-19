from dataclasses import dataclass
from typing import Any

@dataclass
class InputObject:
    type: str
    content: any
    oid: int = 0