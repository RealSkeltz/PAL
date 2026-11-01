"""How events and turns are written out as one-line notes: for the agent's view of what
it hasn't seen yet, and for the recall index."""
from datetime import datetime

from pal.schemas.event import Actor, Event, Language, Observation
from pal.schemas.turn import ConversationTurn, PalTurn, UserTurn


def event_notes(events: list[Event]) -> str:
    """Events as timestamped notes, oldest first; empty when none is worth a note."""
    lines = sorted(((e.at, line) for e in events if (line := event_line(e))), key=lambda x: x[0])
    return "\n".join(f"{at.astimezone():%H:%M:%S} {line}" for at, line in lines)


def event_line(event: Event) -> str | None:
    if isinstance(event, Language):
        if event.actor is Actor.USER:
            return f"user (not to Buddy): {event.text}"
        return f"someone else: {event.text}"
    if isinstance(event, Observation):
        # A look's older frames carry no description of their own; it's on the newest one
        return f"camera: {event.description}" if event.description else None
    return None


def turn_line(turn: ConversationTurn) -> str:
    if isinstance(turn, UserTurn):
        seen = "".join(f" [camera: {o.description}]" for o in turn.attachments if o.description)
        return f"user to Buddy: {turn.utterance.text}{seen}"
    parts = [f"Buddy used {a.tool}({a.args}) -> {a.result}" for a in turn.actions]
    if turn.spoken_text:
        parts.append(f"Buddy: {turn.spoken_text}")
    return " | ".join(parts)


def turn_time(turn: ConversationTurn) -> datetime | None:
    if turn.utterance is not None:
        return turn.utterance.at
    return turn.actions[0].at if isinstance(turn, PalTurn) and turn.actions else None


def ago(delta) -> str:
    minutes = int(delta.total_seconds() // 60)
    return "just now" if minutes < 1 else f"{minutes} min ago"
