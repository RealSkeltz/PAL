"""What has been happening around the user, in order, and how it relates to them.

Each Episode takes one or more events and turns together into a short account
of what happened. It can point to the user's goals it relates to, so progress
can be traced back; untagged episodes are just part of the day's timeline.
Episodes are appended, and only the latest one can still be extended while
it is ongoing, so the timeline itself doesn't drift.
"""
from dataclasses import field
from datetime import datetime
from uuid import UUID, uuid4

from pydantic.dataclasses import dataclass


@dataclass(frozen=True, kw_only=True)
class Episode:
    id: UUID = field(default_factory=uuid4)
    order: int                    # explicit chronological position in the timeline
    at: datetime                  # when its first source event happened
    until: datetime               # when its last source event happened
    what: str
    who: list[str] = field(default_factory=list)
    source_events: list[UUID] = field(default_factory=list)  # events and turns it was made from
    related_goals: list[UUID] = field(default_factory=list)  # ids of goals in GoalMemory


@dataclass(frozen=True, kw_only=True)
class EpisodicMemory:
    episodes: list[Episode] = field(default_factory=list)  # oldest first
    updated_at: datetime | None = None
