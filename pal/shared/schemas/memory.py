from dataclasses import field
from datetime import datetime
from typing import Any

from pydantic.dataclasses import dataclass

from pal.shared.constants.memory import GoalStatus, MemoryType

@dataclass
class Base:
    agent_id: str
    run_id: str
    mem_id: str
    mem_type: MemoryType
    data_type: str | Any
    original_data: Any
    embedding: Any


@dataclass
class Memory:
    base: Base


# Subclasses need their own @dataclass, otherwise their fields aren't part of __init__.
@dataclass
class Event(Memory):
    when: datetime
    what: str
    who: str | None = None
    where: str | None = None


@dataclass
class Fact(Memory):
    """Domain knowledge PAL can look up, e.g. a chunk of a site procedure or building plan."""
    content: str               # "On a fire alarm, evacuate via stairwell B and meet at the north lot."
    source: str | None = None  # where it came from: "site_handbook.pdf, p. 12"


@dataclass
class Goal(Memory):
    """A long-running objective, spanning hours or days."""
    objective: str                                      # "find out why the back gate keeps being unlocked"
    done_when: str | None = None                        # "cause identified and reported to the supervisor"
    status: GoalStatus = GoalStatus.ACTIVE
    deadline: datetime | None = None
    progress: list[str] = field(default_factory=list)  # timestamped notes, appended as things happen


@dataclass
class Task(Goal):
    """A short-term goal, e.g. a watch: "warn me if anyone enters the server room"."""
    trigger: str | None = None    # condition in plain words, for tev to judge against each scene
    goal_id: str | None = None    # parent goal, if this task serves one


@dataclass
class UserPreference(Memory):
    about: str               # topic, so a newer preference can replace an older one: "answer length"
    statement: str           # "prefers brief, to-the-point answers"
    inferred: bool = False   # PAL guessed it rather than the user saying it outright
