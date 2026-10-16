"""Revising the situation: what goes to the model, and how its answer becomes the next Situation.

Each update sends the current situation plus what happened since the last one,
not the whole run, so the picture is built up step by step. The model returns
a structured SituationUpdate; timestamps are kept here, not by the model: a
fact that continues an earlier one keeps its first_seen, and only facts the new
notes support get a fresh last_seen. Facts that go unsupported age, and the
model is told how old they are so it can let them go.
"""
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field

from pal.llm.base import Turn
from pal.response.prompt import about_user
from pal.schemas.event import Actor, Event, Language, Observation
from pal.schemas.situation import Fact, Situation, Source
from pal.schemas.turn import ConversationTurn, PalTurn, UserTurn

PROMPT = """You keep track of the situation around someone wearing Buddy, a voice assistant with a microphone and a camera. The camera faces the way the user looks, so the user is never in the frame.

You get the situation as it stands, then notes on what happened since it was last updated. Return the revised situation:
- place: where the user is
- activity: what is going on right now
- people: who else is around, one entry per person
- goals: what the user is trying to get done

Be conservative. Only include what the notes support, and keep each text short. Mark how something is known: "told" if the user said it to Buddy, "observed" if it was seen or heard directly, "inferred" if you concluded it from indirect clues. A single hint is a guess with low confidence; raise confidence as more notes support it, lower it when notes contradict it.

To continue a fact from the current situation, give its id as previous_id and set supported_now to whether the new notes support it. Drop facts the notes contradict, and facts that have gone long unsupported and no longer seem to hold. Leave place or activity empty when nothing indicates them."""


def system_prompt() -> str:
    about = about_user()
    return PROMPT + (f"\n\nAbout the user: {about} When others say the user's name, they are talking to the user; the user is not one of the people around." if about else "")


class FactUpdate(BaseModel):
    text: str
    source: Literal["told", "observed", "inferred"]
    confidence: float = Field(description="0 to 1")
    previous_id: str | None = Field(description="id of the fact in the current situation this continues, or null if new")
    supported_now: bool = Field(description="whether the new notes support this fact")


class SituationUpdate(BaseModel):
    place: FactUpdate | None
    activity: FactUpdate | None
    people: list[FactUpdate]
    goals: list[FactUpdate]


def request(situation: Situation, events: list[Event], turns: list[ConversationTurn]) -> list[Turn]:
    """The single user message the model gets: current situation, then new notes."""
    text = f"Current situation:\n{_describe(situation)}\n\nNotes since then:\n{_notes(events, turns)}"
    return [Turn(role="user", text=text)]


def apply(previous: Situation, update: SituationUpdate) -> Situation:
    now = datetime.now(timezone.utc)
    earlier = dict(_with_ids(previous))

    def fact(u: FactUpdate | None) -> Fact | None:
        if u is None:
            return None
        before = earlier.get(u.previous_id) if u.previous_id else None
        return Fact(
            text=u.text,
            source=Source(u.source),
            confidence=min(max(u.confidence, 0.0), 1.0),
            first_seen=before.first_seen if before else now,
            last_seen=now if (u.supported_now or before is None) else before.last_seen,
        )

    return Situation(
        place=fact(update.place),
        activity=fact(update.activity),
        people=[fact(u) for u in update.people],
        goals=[fact(u) for u in update.goals],
        updated_at=now,
    )


def summary(situation: Situation) -> str:
    """For people reading the log: one indented line per fact."""
    rows = [(label, f) for label, f in (("place", situation.place), ("activity", situation.activity)) if f]
    rows += [("person", f) for f in situation.people] + [("goal", f) for f in situation.goals]
    if not rows:
        return "    (nothing known yet)"
    return "\n".join(f"    {label:9s} {f.text}  ({f.source.value}, {f.confidence:.1f})" for label, f in rows)


def _with_ids(situation: Situation) -> list[tuple[str, Fact]]:
    facts = []
    if situation.place:
        facts.append(("place", situation.place))
    if situation.activity:
        facts.append(("activity", situation.activity))
    facts += [(f"person{i}", f) for i, f in enumerate(situation.people, 1)]
    facts += [(f"goal{i}", f) for i, f in enumerate(situation.goals, 1)]
    return facts


def _describe(situation: Situation) -> str:
    facts = _with_ids(situation)
    if not facts:
        return "(nothing known yet)"
    now = datetime.now(timezone.utc)
    return "\n".join(
        f"[{fact_id}] {f.text} ({f.source.value}, confidence {f.confidence:.1f}, "
        f"last supported {_ago(now - f.last_seen)})"
        for fact_id, f in facts
    )


def _notes(events: list[Event], turns: list[ConversationTurn]) -> str:
    lines = [(e.at, _event_line(e)) for e in events]
    for turn in turns:
        at = _turn_time(turn)
        if at is not None:
            lines.append((at, _turn_line(turn)))
    lines = [(at, line) for at, line in lines if line]
    if not lines:
        return "(nothing new)"
    return "\n".join(f"{at.astimezone():%H:%M:%S} {line}" for at, line in sorted(lines, key=lambda x: x[0]))


def _event_line(event: Event) -> str | None:
    if isinstance(event, Language):
        if event.actor is Actor.USER:
            return f"user (not to Buddy): {event.text}"
        return f"someone else: {event.text}"
    if isinstance(event, Observation):
        return f"camera: {event.description}"
    return None


def _turn_line(turn: ConversationTurn) -> str:
    if isinstance(turn, UserTurn):
        seen = "".join(f" [camera: {o.description}]" for o in turn.attachments)
        return f"user to Buddy: {turn.utterance.text}{seen}"
    parts = [f"Buddy used {a.tool}({a.args}) -> {a.result}" for a in turn.actions]
    if turn.spoken_text:
        parts.append(f"Buddy: {turn.spoken_text}")
    return " | ".join(parts)


def _turn_time(turn: ConversationTurn) -> datetime | None:
    if turn.utterance is not None:
        return turn.utterance.at
    return turn.actions[0].at if isinstance(turn, PalTurn) and turn.actions else None


def _ago(delta) -> str:
    minutes = int(delta.total_seconds() // 60)
    return "just now" if minutes < 1 else f"{minutes} min ago"
