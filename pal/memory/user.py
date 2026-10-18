"""User-oriented memory: lasting things about the user, and how they like things done."""
from datetime import datetime, timezone

from pydantic import BaseModel

from pal.memory import notes
from pal.memory.notes import FactUpdate
from pal.schemas.user_memory import UserMemory

LABEL = "user memory"

TASK = """You keep a memory about the user of Buddy, a voice assistant they wear with a microphone and a camera.

You get the memory as it stands, then notes on what happened since it was last updated. Return the revised memory:
- about: facts about the user themselves, such as their work, background or circumstances
- people: people in the user's life and how they relate to the user, such as "Mark: the user's brother"
- preferences: how the user likes things, including how they want Buddy to behave (how long its answers should be, what to mention or leave out) and corrections they gave Buddy

Keep only what lasts beyond this moment; where the user is right now and what they are doing is kept elsewhere."""


class Update(BaseModel):
    about: list[FactUpdate]
    people: list[FactUpdate]
    preferences: list[FactUpdate]


def system_prompt() -> str:
    return notes.system_prompt(TASK)


def sections(m: UserMemory):
    return [("about", m.about), ("person", m.people), ("preference", m.preferences)]


def build(current, events, turns, memory):
    return notes.fact_build(sections(current), events, turns)


def summary(current) -> str:
    return notes.summary(sections(current))


def apply(previous: UserMemory, update: Update, context=None) -> UserMemory:
    now = datetime.now(timezone.utc)
    earlier = dict(notes.with_ids(sections(previous)))
    return UserMemory(
        about=[notes.to_fact(u, earlier, now) for u in update.about],
        people=[notes.to_fact(u, earlier, now) for u in update.people],
        preferences=[notes.to_fact(u, earlier, now) for u in update.preferences],
        updated_at=now,
    )
