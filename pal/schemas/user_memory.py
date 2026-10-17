"""What Buddy has learned about the user themselves: lasting things, not the current moment.

Built up the same way as the situation (see situation.py for Fact), but about
the person rather than their surroundings, so it changes slowly.
"""
from dataclasses import field
from datetime import datetime

from pydantic.dataclasses import dataclass

from pal.schemas.situation import Fact


@dataclass(frozen=True, kw_only=True)
class UserMemory:
    about: list[Fact] = field(default_factory=list)        # work, background, circumstances
    people: list[Fact] = field(default_factory=list)       # people in the user's life and how they relate
    preferences: list[Fact] = field(default_factory=list)  # how they like things, including how Buddy should behave
    updated_at: datetime | None = None
