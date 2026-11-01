"""The agent's memory tools: how it keeps its own picture of the user and their world.

Every item in memory has a short key ("u3", "g2") that the agent sees in its
view of memory and edits by. These run quietly: no tool sound, and nothing to
confirm aloud, since keeping memory is the agent's own business.
"""
from pal.tools.base import ToolHandler, tool

SOURCE = {"type": "string", "enum": ["told", "observed", "inferred"],
          "description": "told: the user said it to you; observed: seen or heard directly; inferred: concluded from clues"}
CONFIDENCE = {"type": "number", "description": "0 to 1; a single hint is a guess, raise it as more supports it"}

MEMORY_TOOLS = {"set_situation", "remember", "revise", "forget", "add_goal", "update_goal", "log_episode"}


# Appended to every memory tool's result: a tool result gets the model writing again, and it
# would otherwise confirm the edit aloud
QUIET = " Nothing more to say about this."


class MemoryTools(ToolHandler):

    def dispatch(self, name: str, arguments: dict) -> str:
        result = super().dispatch(name, arguments)
        return result + QUIET if name in MEMORY_TOOLS else result

    @tool(
        "Set the current situation as a whole: where the user is, what is going on, and who else is around. "
        "Leave a part empty when nothing indicates it.",
        params={"place": {"type": "string"}, "activity": {"type": "string"},
                "people": {"type": "array", "items": {"type": "string"}, "description": "one entry per person around"},
                "source": SOURCE, "confidence": CONFIDENCE},
        required=["people", "source", "confidence"],
    )
    def set_situation(self, people: list[str], source: str, confidence: float,
                      place: str | None = None, activity: str | None = None) -> str:
        self.pal.memory.set_situation(place, activity, people, source, confidence)
        return "Situation set."

    @tool(
        "Remember a lasting fact: about the user themselves (work, background, circumstances), a person in "
        "their life and how they relate, or a preference, including how they want you to behave.",
        params={"kind": {"type": "string", "enum": ["about", "person", "preference"]},
                "text": {"type": "string", "description": "short, e.g. 'Mark: the user's brother'"},
                "source": SOURCE, "confidence": CONFIDENCE},
        required=["kind", "text", "source", "confidence"],
    )
    def remember(self, kind: str, text: str, source: str, confidence: float) -> str:
        return f"Remembered as {self.pal.memory.remember(kind, text, source, confidence)}."

    @tool(
        "Change a remembered fact or goal by its key: new wording, or a new confidence when something "
        "supports or contradicts it. It counts as supported now.",
        params={"key": {"type": "string"}, "text": {"type": "string"}, "confidence": CONFIDENCE},
        required=["key"],
    )
    def revise(self, key: str, text: str | None = None, confidence: float | None = None) -> str:
        return f"Revised {key}." if self.pal.memory.revise(key, text, confidence) else f"Nothing is stored as {key}."

    @tool(
        "Forget a remembered fact or goal by its key, when it turned out wrong or no longer holds.",
        params={"key": {"type": "string"}},
        required=["key"],
    )
    def forget(self, key: str) -> str:
        return f"Forgot {key}." if self.pal.memory.forget(key) else f"Nothing is stored as {key}."

    @tool(
        "Add something the user is trying to get done: a long-term goal spanning days or longer, or a task "
        "for today or right now. Only the user's own goals; a request to you isn't one.",
        params={"kind": {"type": "string", "enum": ["long_term", "task"]}, "text": {"type": "string"},
                "source": SOURCE, "confidence": CONFIDENCE},
        required=["kind", "text", "source", "confidence"],
    )
    def add_goal(self, kind: str, text: str, source: str, confidence: float) -> str:
        return f"Added as {self.pal.memory.add_goal(kind, text, source, confidence)}."

    @tool(
        "Update a goal by its key: its status, or a short note on its progress.",
        params={"key": {"type": "string"}, "status": {"type": "string", "enum": ["open", "done", "dropped"]},
                "progress": {"type": "string"}},
        required=["key"],
    )
    def update_goal(self, key: str, status: str | None = None, progress: str | None = None) -> str:
        return f"Updated {key}." if self.pal.memory.update_goal(key, status, progress) else f"No goal is stored as {key}."

    @tool(
        "Add an episode to the timeline: a one-sentence account of something that happened and is worth "
        "remembering later, such as a conversation, a step in a task, someone arriving or a decision. Keep "
        "concrete details: names, numbers, places.",
        params={"what": {"type": "string"},
                "who": {"type": "array", "items": {"type": "string"}, "description": "who was involved besides the user"},
                "goals": {"type": "array", "items": {"type": "string"}, "description": "keys of related goals"}},
        required=["what"],
    )
    def log_episode(self, what: str, who: list[str] | None = None, goals: list[str] | None = None) -> str:
        self.pal.memory.log_episode(what, who or [], goals or [], since=self.pal.turn_started_seeing)
        return "Episode logged."
