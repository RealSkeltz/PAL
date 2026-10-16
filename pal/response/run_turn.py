"""The agent's core: one response to the user, as a stream of events.

`run_turn` talks to the model and runs tools, and nothing else. It does not
speak, play sounds, or write to the history; it yields what happened and the
caller decides what to do with it. A step is one model response: text,
possibly followed by tool calls whose results feed the next step.
"""
import logging
import threading
from dataclasses import dataclass, field, replace
from typing import Iterator

from pal.response.client_turns import to_client_turns
from pal.llm.base import Completed, LLMClient, Turn
from pal.schemas.event import Action, Actor, Role
from pal.schemas.turn import ConversationTurn
from pal.tools.base import ToolHandler

logger = logging.getLogger(__name__)

MAX_STEPS = 5


@dataclass(frozen=True)
class TextChunk:
    text: str


@dataclass(frozen=True)
class ToolStarted:
    action: Action


@dataclass(frozen=True)
class ToolFinished:
    action: Action  # with `result` filled in


@dataclass(frozen=True)
class StepDone:
    text: str
    actions: list[Action] = field(default_factory=list)
    refused: bool = False


AgentEvent = TextChunk | ToolStarted | ToolFinished | StepDone


def run_turn(
    history: list[ConversationTurn],
    system_prompt: str | None,
    client: LLMClient,
    tools: ToolHandler | None = None,
    cancel: threading.Event | None = None,
    max_steps: int = MAX_STEPS,
) -> Iterator[AgentEvent]:
    """Stops early, without a final StepDone, when `cancel` is set."""
    conversation = to_client_turns(history)
    specs = tools.specs if tools else []
    cancelled = lambda: cancel is not None and cancel.is_set()

    for _ in range(max_steps):
        result = Completed()
        for chunk in client.stream(system_prompt, conversation, specs):
            if cancelled():
                return
            if isinstance(chunk, Completed):
                result = chunk
            else:
                yield TextChunk(chunk.text)

        if result.refused:
            yield StepDone(text="", refused=True)
            return

        conversation.append(Turn(role="assistant", text=result.text, tool_calls=result.tool_calls))

        actions = []
        for call in result.tool_calls:
            if cancelled():
                return
            action = Action(actor=Actor.PAL, role=Role.DIRECT, description=f"called {call.name}",
                            tool=call.name, args=call.arguments)
            yield ToolStarted(action)
            output = tools.dispatch(call.name, call.arguments) if tools else "Error: no tool handler configured"
            action = replace(action, result=output)
            yield ToolFinished(action)
            conversation.append(Turn(role="tool", text=output, tool_call_id=call.id))
            actions.append(action)

        yield StepDone(text=result.text, actions=actions)
        if not result.tool_calls:
            return

    logger.warning(f"hit max_steps ({max_steps})")
