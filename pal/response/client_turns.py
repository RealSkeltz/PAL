"""Convert PAL's conversation history into the wire turns a client sends.

History is kept whole; everything that only exists to fit a model's context
(dropping old turns, sending images only for the latest look while it's recent) happens
here, on the way out. A user turn's automatic recall goes out with it every
time, so a later reply sees what an earlier one was based on.
"""
import base64
from datetime import datetime, timezone
from pathlib import Path

from pal.llm.base import ToolCall, Turn
from pal.schemas.event import Action
from pal.schemas.turn import ConversationTurn, Delivery, UserTurn

MAX_TURNS = 20
# Old turns are dropped this many at a time, not one per message: a window that moves every
# turn changes the start of the conversation every time, and prompt caching only reuses an
# unchanged start. So the model sees between MAX_TURNS and MAX_TURNS + DROP_STEP - 1 turns.
DROP_STEP = 10
# The latest look's images stay with the conversation this long, so a follow-up about what was
# just seen ("and the colour?") can still be answered without looking again. After that, or once a
# newer look replaces it, only its description goes along.
RECENT_LOOK_SECONDS = 120


def to_client_turns(history: list[ConversationTurn], max_turns: int = MAX_TURNS) -> list[Turn]:
    # Start on a user turn so a reply never arrives without its question.
    start = max(len(history) - max_turns, 0) // DROP_STEP * DROP_STEP
    while start > 0 and not isinstance(history[start], UserTurn):
        start -= 1
    window = history[start:]

    last_user = max((i for i, t in enumerate(window) if isinstance(t, UserTurn)), default=None)
    last_look = max((i for i, t in enumerate(window) if isinstance(t, UserTurn) and _frames(t)), default=None)
    now = datetime.now(timezone.utc)

    turns = []
    for i, turn in enumerate(window):
        if isinstance(turn, UserTurn):
            frames = _frames(turn)
            current = i == last_user
            shown = i == last_look and (current or (now - frames[-1].at).total_seconds() <= RECENT_LOOK_SECONDS)
            images = [_load_image(o.image_path) for o in frames] if shown else []
            text = _frame_notes(frames, current, shown) + _recall_note(turn.recalled) + turn.utterance.text
            turns.append(Turn(role="user", text=text, images=images))
            continue

        text = turn.spoken_text
        if turn.delivery is Delivery.CANCELLED:
            text += " [interrupted by the user]"
        calls = [ToolCall(name=a.tool, arguments=a.args, id=_call_id(a)) for a in turn.actions]
        turns.append(Turn(role="assistant", text=text, tool_calls=calls))
        turns.extend(Turn(role="tool", text=a.result or "", tool_call_id=_call_id(a)) for a in turn.actions)
    return turns


def _frames(turn: UserTurn) -> list:
    return [o for o in turn.attachments if o.image_path]


def _frame_notes(frames: list, current: bool, shown: bool) -> str:
    """Say when the attached images were seen, so a new view isn't mistaken for a contradiction of
    an old one. Worded as seeing, not as frames: the reply should talk the same way."""
    if not frames:
        return ""
    times = [f"{o.at.astimezone():%H:%M:%S.%f}"[:-5] for o in frames]
    when = times[0] if len(times) == 1 else f"{times[0]} to {times[-1]}, oldest first"
    if shown and current:
        return f"[What you see as the user says this ({when}). Not a request to describe it.]\n"
    if shown:
        return f"[What you saw when the user said this ({when}); still with you for follow-up questions.]\n"
    seen = " ".join(dict.fromkeys(o.description for o in frames if o.description))
    return (f"[You saw what was in front of the user when they said this ({when})"
            + (f": {seen}" if seen else "") + ". You no longer see it, but your answer was based on seeing it.]\n")


def _recall_note(recalled: str) -> str:
    if not recalled:
        return ""
    return f"[Found automatically for this message; use it only if it helps.\n{recalled}]\n"


def _call_id(action: Action) -> str:
    return f"toolu_{action.id.hex}"


def _load_image(path: str) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode("utf-8")
