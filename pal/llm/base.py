"""Provider-neutral LLM interface.

Everything the agent loop needs from a model lives behind `LLMClient`. Backends
translate to and from their own wire format, so nothing above this layer knows
whether it is talking to a local ollama model or the Claude API.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterator


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
        system: str | None,
        conversation: list[Turn],
        tools: list[ToolSpec] | None = None,
    ) -> Iterator[TextDelta | Completed]:
        """Yield TextDelta as text arrives, then exactly one Completed at the end."""

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
