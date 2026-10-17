"""What the user is trying to get done: long-term goals and short-term tasks.

Each is a Fact (see situation.py) with a status, so finished and abandoned ones
stay known for a while instead of silently disappearing.
"""
from dataclasses import field
from datetime import datetime
from enum import StrEnum

from pydantic.dataclasses import dataclass

from pal.schemas.situation import Fact


class Status(StrEnum):
    OPEN = "open"
    DONE = "done"
    DROPPED = "dropped"


@dataclass(frozen=True, kw_only=True)
class Goal(Fact):
    status: Status = Status.OPEN
    progress: str | None = None


@dataclass(frozen=True, kw_only=True)
class GoalMemory:
    long_term: list[Goal] = field(default_factory=list)  # spanning days or longer
    tasks: list[Goal] = field(default_factory=list)      # concrete things for today or right now
    updated_at: datetime | None = None
