"""What Buddy has gathered about the user's surroundings over a run.

A small, distilled picture built up from many events: where the user is, who
is around, and what is going on. Each item says
how it is known, how sure Buddy is, and when it was last supported, so a single
overheard remark stays a tentative guess, and items nothing supports anymore
can fade.

It is replaced as a whole on every update, so it is frozen like an event.
"""
from dataclasses import field
from datetime import datetime, timezone
from enum import StrEnum
from typing import Annotated

from pydantic import Field
from pydantic.dataclasses import dataclass


class Source(StrEnum):
    TOLD = "told"          # the user said it to Buddy
    OBSERVED = "observed"  # seen or heard directly
    INFERRED = "inferred"  # concluded from indirect evidence


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True, kw_only=True)
class Fact:
    text: str
    source: Source
    confidence: Annotated[float, Field(ge=0, le=1)]
    first_seen: datetime = field(default_factory=_now)
    last_seen: datetime = field(default_factory=_now)  # last time anything supported it


@dataclass(frozen=True, kw_only=True)
class Situation:
    place: Fact | None = None
    activity: Fact | None = None
    people: list[Fact] = field(default_factory=list)  # who else is around right now
    updated_at: datetime | None = None                # None until the first update
