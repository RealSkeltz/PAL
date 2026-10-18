"""What the user is trying to get done: long-term goals and short-term tasks.

Each goal keeps a stable id across updates, so episodes can point to the goals
they relate to. A status keeps finished and abandoned ones known for a while
instead of silently disappearing.
"""
from dataclasses import field
from datetime import datetime
from enum import StrEnum
from typing import Annotated
from uuid import UUID, uuid4

from pydantic import Field
from pydantic.dataclasses import dataclass

from pal.schemas.situation import Source, _now


class Status(StrEnum):
    OPEN = "open"
    DONE = "done"
    DROPPED = "dropped"


@dataclass(frozen=True, kw_only=True)
class Goal:
    id: UUID = field(default_factory=uuid4)
    text: str
    source: Source
    confidence: Annotated[float, Field(ge=0, le=1)]
    status: Status = Status.OPEN
    progress: str | None = None
    first_seen: datetime = field(default_factory=_now)
    last_seen: datetime = field(default_factory=_now)  # last time anything supported it


@dataclass(frozen=True, kw_only=True)
class GoalMemory:
    long_term: list[Goal] = field(default_factory=list)  # spanning days or longer
    tasks: list[Goal] = field(default_factory=list)      # concrete things for today or right now
    updated_at: datetime | None = None
