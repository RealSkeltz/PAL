"""The building block for everything PAL takes in.

    Event
    ├── Language          something said or written
    │   ├── Spoken
    │   └── Written
    ├── Observation       what the world looks like (state)
    └── Action            something done (change)

Observation and Action split on state versus change: "keys on the kitchen
table" is an observation, "the user put the keys on the table" is an action.
Actions cover both what PAL perceives others doing and what PAL itself does;
`actor` says who did it, and PAL's own tool calls also fill `tool` and `args`.

Every event also carries a `role`: DIRECT input is addressed to PAL and may
need a response, CONTEXT is background PAL should know about but not act on.
This is a field rather than a subclass because it cuts across the tree: the
same spoken sentence is direct when said to PAL and context when said to
someone else.

Events are frozen. When a later stage learns more (e.g. the response gate
decides an utterance was for PAL), it makes a new event with
`dataclasses.replace(event, role=Role.DIRECT)`.
"""
from dataclasses import field
from datetime import datetime, timezone
from enum import StrEnum
from typing import Annotated, Any, Literal
from uuid import UUID, uuid4

from pydantic import Field, TypeAdapter
from pydantic.dataclasses import dataclass


class Role(StrEnum):
    DIRECT = "direct"
    CONTEXT = "context"


class Actor(StrEnum):
    USER = "user"    # the person wearing PAL
    OTHER = "other"  # anyone else
    PAL = "pal"


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True, kw_only=True)
class Event:
    actor: Actor
    role: Role
    id: UUID = field(default_factory=uuid4)
    at: datetime = field(default_factory=_now)


@dataclass(frozen=True, kw_only=True)
class Language(Event):
    text: str


@dataclass(frozen=True, kw_only=True)
class Spoken(Language):
    kind: Literal["spoken"] = "spoken"
    duration_s: float | None = None
    speaker_score: float | None = None  # voiceprint similarity behind `actor`


@dataclass(frozen=True, kw_only=True)
class Written(Language):
    kind: Literal["written"] = "written"


@dataclass(frozen=True, kw_only=True)
class Observation(Event):
    kind: Literal["observation"] = "observation"
    description: str
    label: str | None = None         # detector class, e.g. "keys"
    confidence: float | None = None  # detector confidence for `label`
    image_path: str | None = None    # the frame on disk; arrays don't belong in an event


@dataclass(frozen=True, kw_only=True)
class Action(Event):
    kind: Literal["action"] = "action"
    description: str
    confidence: float | None = None
    tool: str | None = None  # set when PAL itself acted through a tool
    args: dict[str, Any] = field(default_factory=dict)
    result: str | None = None  # the tool's output, filled in once it returns


# Any concrete event; `kind` tells pydantic which class to build when parsing.
AnyEvent = Annotated[Spoken | Written | Observation | Action, Field(discriminator="kind")]
event_adapter: TypeAdapter[AnyEvent] = TypeAdapter(AnyEvent)
