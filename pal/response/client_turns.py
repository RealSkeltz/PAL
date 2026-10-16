"""Convert PAL's conversation history into the wire turns a client sends.

History is kept whole; everything that only exists to fit a model's context
(dropping old turns, sending images for the latest user turn only) happens
here, on the way out.
"""
import base64
from pathlib import Path

from pal.llm.base import ToolCall, Turn
from pal.schemas.event import Action
from pal.schemas.turn import ConversationTurn, Delivery, UserTurn

MAX_TURNS = 20


def to_client_turns(history: list[ConversationTurn], max_turns: int = MAX_TURNS) -> list[Turn]:
    # Start on a user turn so a reply never arrives without its question.
    start = max(len(history) - max_turns, 0)
    while start > 0 and not isinstance(history[start], UserTurn):
        start -= 1
    window = history[start:]

    last_user = max((i for i, t in enumerate(window) if isinstance(t, UserTurn)), default=None)

    turns = []
    for i, turn in enumerate(window):
        if isinstance(turn, UserTurn):
            images = [_load_image(o.image_path) for o in turn.attachments if o.image_path] if i == last_user else []
            turns.append(Turn(role="user", text=turn.utterance.text, images=images))
            continue

        text = turn.spoken_text
        if turn.delivery is Delivery.CANCELLED:
            text += " [interrupted by the user]"
        calls = [ToolCall(name=a.tool, arguments=a.args, id=_call_id(a)) for a in turn.actions]
        turns.append(Turn(role="assistant", text=text, tool_calls=calls))
        turns.extend(Turn(role="tool", text=a.result or "", tool_call_id=_call_id(a)) for a in turn.actions)
    return turns


def _call_id(action: Action) -> str:
    return f"toolu_{action.id.hex}"


def _load_image(path: str) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode("utf-8")
