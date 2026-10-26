"""Provider-neutral LLM interface.

Everything the agent loop needs from a model lives behind `LLMClient`. Backends
translate to and from their own wire format, so nothing above this layer knows
whether it is talking to a local ollama model or the Claude API.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterator, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


@dataclass
class ToolSpec:
    """A tool, described independently of any provider's schema dialect."""
    name: str
    description: str
    properties: dict = field(default_factory=dict)   # JSON Schema for the arguments
    required: list[str] = field(default_factory=list)


@dataclass
class ToolCall:
    name: str
    arguments: dict = field(default_factory=dict)
    id: str = ""              # Claude pairs results to calls by id; ollama doesn't


@dataclass
class Turn:
    """One entry in a conversation."""
    role: str                                          # user | assistant | tool
    text: str = ""
    images: list[str] = field(default_factory=list)    # base64-encoded JPEG
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str = ""                             # set when role == "tool"


# --- Stream events ---------------------------------------------------------

@dataclass
class TextDelta:
    """A fragment of assistant text, as it arrives."""
    text: str


@dataclass
class Completed:
    """End of a turn: the full text plus any tool calls the model made."""
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    refused: bool = False     # model declined on policy grounds; text may be empty


class LLMClient(ABC):
    """Streams one assistant turn at a time."""

    @abstractmethod
    def stream(
        self,
        system: str | list[str] | None,
        conversation: list[Turn],
        tools: list[ToolSpec] | None = None,
    ) -> Iterator[TextDelta | Completed]:
        """Yield TextDelta as text arrives, then exactly one Completed at the end. `system` may
        be a list of parts, the most stable first, so a client that caches can cache them apart."""

    def parse(self, system: str | None, conversation: list[Turn], output: type[T]) -> T:
        """One reply, constrained to the shape of `output` and returned as an instance of it."""
        raise NotImplementedError(f"{type(self).__name__} has no structured output")

    def warm_up(self) -> bool:
        """Load the model now so the first request isn't slow. True if there was anything to load."""
        return False

    def release(self):
        """Free the model's memory. Only local backends need this."""

    def complete(
        self,
        system: str | None,
        conversation: list[Turn],
        tools: list[ToolSpec] | None = None,
    ) -> Completed:
        """Drain the stream and return only the final result."""
        result = Completed()
        for event in self.stream(system, conversation, tools):
            if isinstance(event, Completed):
                result = event
        return result


def system_text(system: str | list[str] | None) -> str | None:
    """A system prompt given as parts, as one string, for clients that take only that."""
    if system is None or isinstance(system, str):
        return system
    return "\n\n".join(p for p in system if p) or None
