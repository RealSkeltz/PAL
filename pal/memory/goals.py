"""Goal-oriented memory: the user's long-term goals and short-term tasks, with their status."""
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field

from pal.memory import notes
from pal.memory.notes import FactUpdate
from pal.schemas.goals import Goal, GoalMemory, Status

LABEL = "goals"

TASK = """You keep track of what the user of Buddy, a voice assistant they wear, is trying to get done.

You get the goals as they stand, then notes on what happened since they were last updated. Return the revised goals, in two kinds:
- long_term: goals that span days or longer, such as finishing a project
- tasks: concrete things to do today or right now, such as finishing a delivery check before two

Give each a status: "open", "done" when the notes show it was completed, or "dropped" when it was abandoned or no longer matters. Add a short progress note when the notes show partial progress. Keep done and dropped items for a while so they aren't forgotten, then drop them.

Only the user's own goals: a request Buddy carried out (setting a timer, saving a note) is not a goal unless it serves one."""


class GoalUpdate(FactUpdate):
    status: Literal["open", "done", "dropped"]
    progress: str | None = Field(description="short progress note, or null")


class Update(BaseModel):
    long_term: list[GoalUpdate]
    tasks: list[GoalUpdate]


def system_prompt() -> str:
    return notes.system_prompt(TASK)


def sections(g: GoalMemory):
    return [("goal", g.long_term), ("task", g.tasks)]


def build(current, events, turns, memory):
    return notes.fact_build(sections(current), events, turns)


def summary(current) -> str:
    return notes.summary(sections(current))


def apply(previous: GoalMemory, update: Update, context=None) -> GoalMemory:
    now = datetime.now(timezone.utc)
    earlier = dict(notes.with_ids(sections(previous)))
    goal = lambda u: notes.to_fact(u, earlier, now, cls=Goal, status=Status(u.status), progress=u.progress)
    return GoalMemory(long_term=[goal(u) for u in update.long_term], tasks=[goal(u) for u in update.tasks], updated_at=now)
