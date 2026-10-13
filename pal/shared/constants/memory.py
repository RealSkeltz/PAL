from enum import StrEnum, auto


class MemoryType(StrEnum):
    # auto() on a StrEnum gives the lowercased member name, e.g. "goal_oriented"
    SEMANTIC = auto()
    EPISODIC = auto()
    GOAL_ORIENTED = auto()
    USER_ORIENTED = auto()


MEMORY_TYPES = tuple(MemoryType)


class GoalStatus(StrEnum):
    ACTIVE = auto()
    PAUSED = auto()
    DONE = auto()
    ABANDONED = auto()
