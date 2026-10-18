"""What the three memory updaters share: how new events are written out for the model,
how current facts are shown with ids, the common rules, and how answers become Facts.

Each update sends the current state plus what happened since the last update,
not the whole run, so memory is built up step by step. Timestamps are kept
here, not by the model: a fact that continues an earlier one keeps its
first_seen, and only facts the new notes support get a fresh last_seen. Facts
that go unsupported age, and the model is told how old they are so it can let
them go.
"""
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field

from pal.llm.base import Turn
from pal.response.prompt import about_user
from pal.schemas.event import Actor, Event, Language, Observation
from pal.schemas.goals import Goal
from pal.schemas.situation import Fact, Source
from pal.schemas.turn import ConversationTurn, PalTurn, UserTurn

RULES = """Describe the user's world, not their use of Buddy: "camera" notes are simply what is in front of the user, and asking Buddy something is never itself a goal or a fact worth keeping.

Be conservative. Only include what the notes support, and keep each text short. Mark how something is known: "told" if the user said it to Buddy, "observed" if it was seen or heard directly, "inferred" if you concluded it from indirect clues. A single hint is a guess with low confidence; raise confidence as more notes support it, lower it when notes contradict it.

To continue an item from the current state, give its id as previous_id and set supported_now to whether the new notes support it. Drop items the notes contradict, and items that have gone long unsupported and no longer seem to hold."""


def system_prompt(task: str) -> str:
    about = about_user()
    who = (f"\n\nAbout the user: {about} When others say the user's name, they are talking to the user; "
           "the user is not one of the people around.") if about else ""
    return f"{task}\n\n{RULES}{who}"


class FactUpdate(BaseModel):
    text: str
    source: Literal["told", "observed", "inferred"]
    confidence: float = Field(description="0 to 1")
    previous_id: str | None = Field(description="id of the item in the current state this continues, or null if new")
    supported_now: bool = Field(description="whether the new notes support this item")


def request(current: str, events: list[Event], turns: list[ConversationTurn]) -> list[Turn]:
    """The single user message an updater gets: current state, then new notes."""
    return [Turn(role="user", text=f"Current state:\n{current}\n\nNotes since then:\n{_notes(events, turns)}")]


def with_ids(sections: list[tuple[str, list[Fact]]]) -> list[tuple[str, Fact]]:
    """("person", [a, b]) -> [("person1", a), ("person2", b)]; ids the model refers back to."""
    return [(f"{prefix}{i}", f) for prefix, facts in sections for i, f in enumerate(facts, 1)]


def describe(sections: list[tuple[str, list[Fact]]]) -> str:
    facts = with_ids(sections)
    if not facts:
        return "(nothing known yet)"
    now = datetime.now(timezone.utc)
    return "\n".join(f"[{fact_id}] {f.text}{_goal_detail(f)} ({f.source.value}, confidence {f.confidence:.1f}, "
                     f"last supported {_ago(now - f.last_seen)})" for fact_id, f in facts)


def to_fact(u: FactUpdate | None, earlier: dict[str, Fact], now: datetime, cls=Fact, **extra) -> Fact | None:
    if u is None:
        return None
    before = earlier.get(u.previous_id) if u.previous_id else None
    if before is not None and hasattr(before, "id"):
        extra.setdefault("id", before.id)  # e.g. a goal keeps its id, so episodes can keep pointing to it
    return cls(
        text=u.text,
        source=Source(u.source),
        confidence=min(max(u.confidence, 0.0), 1.0),
        first_seen=before.first_seen if before else now,
        last_seen=now if (u.supported_now or before is None) else before.last_seen,
        **extra,
    )


def summary(sections: list[tuple[str, list[Fact]]]) -> str:
    """For people reading the log: one indented line per fact."""
    rows = [(label, f) for label, facts in sections for f in facts]
    if not rows:
        return "    (nothing known yet)"
    return "\n".join(f"    {label:11s} {f.text}{_goal_detail(f)}  ({f.source.value}, {f.confidence:.1f})" for label, f in rows)


def fact_build(sections: list[tuple[str, list[Fact]]], events: list[Event], turns: list[ConversationTurn]):
    """`build` for the fact-based memories: current facts with ids, then new notes. No extra context."""
    return request(describe(sections), events, turns), None


def numbered_notes(events: list[Event], turns: list[ConversationTurn]) -> tuple[str, dict[str, tuple]]:
    """Notes with an id per line ("n1", "n2", ...), and what each id stands for: (source id, time)."""
    items = [(e.at, e.id, event_line(e)) for e in events]
    items += [(turn_time(t), t.id, turn_line(t)) for t in turns if turn_time(t) is not None]
    items = sorted((x for x in items if x[2]), key=lambda x: x[0])
    ids = {f"n{i}": (source, at) for i, (at, source, _) in enumerate(items, 1)}
    text = "\n".join(f"[n{i}] {at.astimezone():%H:%M:%S} {line}" for i, (at, _, line) in enumerate(items, 1))
    return text or "(nothing new)", ids


def _goal_detail(f: Fact) -> str:
    if not isinstance(f, Goal):
        return ""
    return f" [{f.status.value}]" + (f" progress: {f.progress}" if f.progress else "")


def _notes(events: list[Event], turns: list[ConversationTurn]) -> str:
    lines = [(e.at, event_line(e)) for e in events]
    for turn in turns:
        at = turn_time(turn)
        if at is not None:
            lines.append((at, turn_line(turn)))
    lines = [(at, line) for at, line in lines if line]
    if not lines:
        return "(nothing new)"
    return "\n".join(f"{at.astimezone():%H:%M:%S} {line}" for at, line in sorted(lines, key=lambda x: x[0]))


def event_line(event: Event) -> str | None:
    if isinstance(event, Language):
        if event.actor is Actor.USER:
            return f"user (not to Buddy): {event.text}"
        return f"someone else: {event.text}"
    if isinstance(event, Observation):
        return f"camera: {event.description}"
    return None


def turn_line(turn: ConversationTurn) -> str:
    if isinstance(turn, UserTurn):
        seen = "".join(f" [camera: {o.description}]" for o in turn.attachments)
        return f"user to Buddy: {turn.utterance.text}{seen}"
    parts = [f"Buddy used {a.tool}({a.args}) -> {a.result}" for a in turn.actions]
    if turn.spoken_text:
        parts.append(f"Buddy: {turn.spoken_text}")
    return " | ".join(parts)


def turn_time(turn: ConversationTurn) -> datetime | None:
    if turn.utterance is not None:
        return turn.utterance.at
    return turn.actions[0].at if isinstance(turn, PalTurn) and turn.actions else None


def _ago(delta) -> str:
    minutes = int(delta.total_seconds() // 60)
    return "just now" if minutes < 1 else f"{minutes} min ago"
