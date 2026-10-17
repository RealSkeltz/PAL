"""Situation: where the user is, what is going on, and who is around, right now."""
from datetime import datetime, timezone

from pydantic import BaseModel

from pal.memory import notes
from pal.memory.notes import FactUpdate
from pal.schemas.situation import Situation

LABEL = "situation"

TASK = """You keep track of the situation around someone wearing Buddy, a voice assistant with a microphone and a camera. The camera faces the way the user looks, so the user is never in the frame.

You get the situation as it stands, then notes on what happened since it was last updated. Return the revised situation:
- place: where the user is
- activity: what is going on right now
- people: who else is around right now, one entry per person

Leave place or activity empty when nothing indicates them. Lasting facts about the user and their goals are kept elsewhere; this is only about the present moment."""


class Update(BaseModel):
    place: FactUpdate | None
    activity: FactUpdate | None
    people: list[FactUpdate]


def system_prompt() -> str:
    return notes.system_prompt(TASK)


def sections(s: Situation):
    return [("place", [s.place] if s.place else []), ("activity", [s.activity] if s.activity else []), ("person", s.people)]


def apply(previous: Situation, update: Update) -> Situation:
    now = datetime.now(timezone.utc)
    earlier = dict(notes.with_ids(sections(previous)))
    return Situation(
        place=notes.to_fact(update.place, earlier, now),
        activity=notes.to_fact(update.activity, earlier, now),
        people=[notes.to_fact(u, earlier, now) for u in update.people],
        updated_at=now,
    )
