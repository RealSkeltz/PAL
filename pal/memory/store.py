"""Everything Buddy knows so far, from raw to distilled.

    context       every event that wasn't a conversation turn, as it arrived
    history       the conversation between the user and Buddy
    situation     where the user is, what is going on, who is around (right now)
    user_memory   lasting facts about the user, people in their life, their preferences
    goal_memory   the user's long-term goals and short-term tasks, with status
    episodic_memory  a timeline of what happened, tied to the goals it relates to
    knowledge     semantic memory: reference documents, chunked and searchable

The router writes events and turns in. The distilled memories (situation, user,
goals, episodes) are kept by the agent itself, through its memory tools: it sees
them in `view()`, with a short key per item, and edits single items by key. Code
keeps the bookkeeping: keys, and when an item was first and last supported.

`seen` marks how far into the context the agent has looked: every agent turn
gets the events after it (`unseen()`), and moves it on once the turn is done.
Events left unseen are what a reflect turn is for.

Everything is also indexed for recall (`index_loop` embeds new entries in the
background); the gate reads `gate_brief()`.
"""
import logging
import time
from dataclasses import replace
from datetime import datetime, timezone
from itertools import count

from pal.log import record
from pal.memory import notes
from pal.memory.knowledge import Knowledge
from pal.memory.recall import Entry, Hit, RecallIndex, format_hits
from pal.llm.embedding_client import EmbeddingClient
from pal.schemas.episode import Episode, EpisodicMemory
from pal.schemas.event import Event, Observation
from pal.schemas.goals import Goal, GoalMemory, Status
from pal.schemas.situation import Fact, Situation, Source
from pal.schemas.turn import ConversationTurn, Delivery, Response, UserTurn
from pal.schemas.user_memory import UserMemory

logger = logging.getLogger(__name__)

UNSURE = 0.6          # below this, the view marks an item as a guess
GATE_MIN = 0.5        # below this, an item is left out of the gate's brief
EPISODES_IN_VIEW = 8  # latest episodes the agent sees
RECENT_TURNS = 20     # the conversation the response model already sees; not worth recalling

USER_KINDS = {"about": "about", "person": "people", "preference": "preferences"}  # remember(kind) -> UserMemory field
GOAL_KINDS = {"long_term": "long_term", "task": "tasks"}


class Memory:
    def __init__(self, embedder: EmbeddingClient | None = None):
        self.recall_index = RecallIndex(embedder)
        self.knowledge = Knowledge(embedder)  # semantic memory: reference material, loaded by the agent

        self.context: list[Event] = []
        self.history: list[ConversationTurn] = []
        self.situation = Situation()
        self.user_memory = UserMemory()
        self.goal_memory = GoalMemory()
        self.episodic_memory = EpisodicMemory()

        self.seen = 0  # context[:seen] has been in an agent turn's input
        self._keys = count(1)

    def add_event(self, event: Event):
        self.context.append(event)

    def add_turn(self, turn: ConversationTurn):
        self.history.append(turn)

    def describe_look(self, frames: list[Observation], description: str):
        """Fill in what a look showed, once it has been put into words. It goes on the look's newest
        frame, both in the context and on the turn that asked."""
        described = replace(frames[-1], description=description)
        for i, e in enumerate(self.context):  # in place: the router may be appending meanwhile
            if e.id == described.id:
                self.context[i] = described
                break
        for turn in self.history:
            if isinstance(turn, UserTurn) and any(o.id == described.id for o in turn.attachments):
                turn.attachments = [described if o.id == described.id else o for o in turn.attachments]

    def pending(self) -> list[UserTurn]:
        """User turns nobody has started answering yet."""
        return [t for t in self.history if isinstance(t, UserTurn) and t.response is Response.PENDING]

    # --- What the agent hasn't seen yet ---------------------------------------

    def unseen(self) -> tuple[int, str]:
        """The events after `seen` written out as notes, and where they end: pass that to
        `mark_seen` once the turn that saw them is done. Empty when nothing new is worth a note."""
        end = len(self.context)
        return end, notes.event_notes(self.context[self.seen:end])

    def mark_seen(self, end: int):
        self.seen = max(self.seen, end)

    def unseen_span(self) -> tuple[float, float] | None:
        """When the first and the latest unseen event worth a note happened, in epoch seconds."""
        times = [e.at.timestamp() for e in self.context[self.seen:] if notes.event_line(e)]
        return (min(times), max(times)) if times else None

    # --- What the agent sees -----------------------------------------------------

    def view(self) -> str:
        """Everything known, each item with its key, for the agent to read and edit."""
        s, u, g = self.situation, self.user_memory, self.goal_memory
        now = datetime.now(timezone.utc)
        item = lambda f: f"[{f.key}] {_item(f)} ({f.source.value}, last supported {notes.ago(now - f.last_seen)})"
        lines = []
        add = lambda label, facts: lines.append(f"{label}:\n" + "\n".join(f"  {item(f)}" for f in facts)) if facts else None
        situation = [f"{label}: {_item(f)}" for label, f in (("place", s.place), ("going on", s.activity)) if f]
        if s.people:
            situation.append("people around: " + "; ".join(_item(f) for f in s.people))
        lines.append("Situation (set as a whole): " + (", ".join(situation) if situation else "unknown")
                     + (f", set {notes.ago(now - s.updated_at)}" if s.updated_at else ""))
        add("About the user", u.about)
        add("People in the user's life", u.people)
        add("The user's preferences", u.preferences)
        add("Long-term goals", g.long_term)
        add("Current tasks", g.tasks)
        episodes = self.episodic_memory.episodes[-EPISODES_IN_VIEW:]
        if episodes:
            lines.append("Latest episodes, in order:\n" + "\n".join(
                f"  {e.at.astimezone():%H:%M} {e.what}" for e in episodes))
        return "\n".join(lines)

    # --- Edits (the agent's memory tools) -----------------------------------------

    def set_situation(self, place: str | None, activity: str | None, people: list[str],
                      source: str, confidence: float):
        """Replace the situation. An item said the same way as before keeps when it was first seen."""
        before = {f.text: f for f in [self.situation.place, self.situation.activity, *self.situation.people] if f}
        fact = lambda text: text and self._fact(text, source, confidence, before.get(text))
        self.situation = Situation(place=fact(place) or None, activity=fact(activity) or None,
                                   people=[fact(p) for p in people if p], updated_at=_now())
        record("situation", self.situation)

    def remember(self, kind: str, text: str, source: str, confidence: float) -> str:
        """Add a lasting fact about the user: `kind` is "about", "person" or "preference". Returns its key."""
        field = USER_KINDS[kind]
        fact = self._fact(text, source, confidence, key=f"u{next(self._keys)}")
        self.user_memory = replace(self.user_memory, **{field: [*getattr(self.user_memory, field), fact]},
                                   updated_at=_now())
        record("user_memory", self.user_memory)
        return fact.key

    def revise(self, key: str, text: str | None = None, confidence: float | None = None) -> bool:
        """Change a user fact or goal; it counts as supported now. False if `key` isn't known."""
        changes = {k: v for k, v in (("text", text), ("confidence", confidence)) if v is not None}
        return self._edit(key, lambda f: replace(f, **changes, last_seen=_now()))

    def forget(self, key: str) -> bool:
        return self._edit(key, lambda f: None)

    def add_goal(self, kind: str, text: str, source: str, confidence: float) -> str:
        """Add a goal: `kind` is "long_term" or "task". Returns its key."""
        field = GOAL_KINDS[kind]
        goal = Goal(key=f"g{next(self._keys)}", text=text, source=Source(source), confidence=_clamp(confidence))
        self.goal_memory = replace(self.goal_memory, **{field: [*getattr(self.goal_memory, field), goal]},
                                   updated_at=_now())
        record("goal_memory", self.goal_memory)
        return goal.key

    def update_goal(self, key: str, status: str | None = None, progress: str | None = None) -> bool:
        changes = {k: v for k, v in (("status", status and Status(status)), ("progress", progress)) if v is not None}
        return self._edit(key, lambda g: replace(g, **changes, last_seen=_now()))

    def log_episode(self, what: str, who: list[str], goals: list[str], since: float | None = None):
        """Add an episode to the timeline, from `since` (epoch seconds) to now, linked to goals by key."""
        all_goals = self.goal_memory.long_term + self.goal_memory.tasks
        episodes = self.episodic_memory.episodes
        now = _now()
        episode = Episode(order=len(episodes) + 1, what=what, who=who,
                          at=datetime.fromtimestamp(since, timezone.utc) if since else now, until=now,
                          related_goals=[g.id for g in all_goals if g.key in goals])
        self.episodic_memory = EpisodicMemory(episodes=[*episodes, episode], updated_at=now)
        record("episodic_memory", self.episodic_memory)

    def _fact(self, text: str, source: str, confidence: float, before: Fact | None = None, key: str = "") -> Fact:
        return Fact(key=before.key if before else key, text=text, source=Source(source), confidence=_clamp(confidence),
                    first_seen=before.first_seen if before else _now())

    def _edit(self, key: str, change) -> bool:
        """Apply `change` (item -> item, or None to drop it) to the user fact or goal with `key`."""
        for attr, fields in (("user_memory", USER_KINDS.values()), ("goal_memory", GOAL_KINDS.values())):
            memory = getattr(self, attr)
            for field in fields:
                items = getattr(memory, field)
                if any(i.key == key for i in items):
                    kept = [change(i) if i.key == key else i for i in items]
                    setattr(self, attr, replace(memory, **{field: [i for i in kept if i is not None]}, updated_at=_now()))
                    record(attr, getattr(self, attr))
                    return True
        return False

    # --- Recall ---------------------------------------------------------------

    def index_loop(self, every_seconds: float = 2.0):
        """Run forever: index new events, turns and episodes, and embed them. Meant for its own thread."""
        while True:
            try:
                self._sync_index()
            except Exception:
                logger.exception("indexing for recall failed")
            time.sleep(every_seconds)

    def recall(self, query: str, k: int = 5, include_recent: bool = False) -> list[Hit]:
        """Earlier events, turns and episodes matching `query`, by keyword and meaning. By default
        leaves out the recent conversation, which the response model already has."""
        self._sync_index()
        recent = set() if include_recent else {t.id for t in list(self.history)[-RECENT_TURNS:]}
        return self.recall_index.search(query, k=k, exclude=recent)

    def recall_text(self, query: str, k: int = 8) -> str:
        return format_hits(self.recall(query, k=k))

    def _sync_index(self):
        entries = [Entry(e.id, e.at, "event", line) for e in list(self.context) if (line := notes.event_line(e))]
        # A reply cut off before its first word has no text; one empty input fails the whole embedding batch.
        entries += [Entry(t.id, at, "turn", line) for t in list(self.history)
                    if not _in_progress(t) and (at := notes.turn_time(t)) is not None and (line := notes.turn_line(t))]
        entries += [Entry(e.id, e.at, "episode", f"episode: {e.what}" + (f" (with {', '.join(e.who)})" if e.who else ""))
                    for e in self.episodic_memory.episodes]
        self.recall_index.add(entries)
        self.recall_index.embed_pending()

    def gate_brief(self) -> str | None:
        """For the gate: a few words on the present moment. Short, since the gate reads every token."""
        s = self.situation
        parts = [f.text for f in (s.place, s.activity) if f and f.confidence >= GATE_MIN]
        people = [f.text for f in s.people if f.confidence >= GATE_MIN]
        if people:
            parts.append("around: " + ", ".join(people))
        return "; ".join(parts)[:240] or None


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _clamp(confidence: float) -> float:
    return min(max(float(confidence), 0.0), 1.0)


def _item(f) -> str:
    return f.text + (" (unsure)" if f.confidence < UNSURE else "") + \
        (f" [{f.status.value}]" if isinstance(f, Goal) else "") + \
        (f", progress: {f.progress}" if getattr(f, "progress", None) else "")


def _in_progress(turn: ConversationTurn) -> bool:
    """Turns whose outcome isn't settled yet; they'll be taken in next time."""
    if isinstance(turn, UserTurn):
        return turn.response in (Response.PENDING, Response.RESPONDING)
    return turn.delivery is Delivery.IN_PROGRESS
