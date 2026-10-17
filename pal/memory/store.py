"""Everything Buddy knows so far, from raw to distilled.

    context       every event that wasn't a conversation turn, as it arrived
    history       the conversation between the user and Buddy
    situation     where the user is, what is going on, who is around (right now)
    user_memory   lasting facts about the user, people in their life, their preferences
    goal_memory   the user's long-term goals and short-term tasks, with status

The router writes events and turns in. The responder reads the history and
`background()`; the gate reads `gate_brief()`. The three distilled memories are
revised in the background (`watch`) once the user and Buddy haven't talked for
a while, so they never compete with a reply. Other people talking doesn't count
as talking to Buddy; their remarks are simply batched into the next update.
"""
import logging
import time
from typing import Callable

from pal.llm.base import LLMClient
from pal.log import record
from pal.memory import goals as goal_updater
from pal.memory import notes
from pal.memory import situation as situation_updater
from pal.memory import user as user_updater
from pal.schemas.event import Event
from pal.schemas.goals import GoalMemory, Status
from pal.schemas.situation import Situation
from pal.schemas.turn import ConversationTurn, Delivery, Response, UserTurn
from pal.schemas.user_memory import UserMemory

logger = logging.getLogger(__name__)

QUIET_SECONDS = 30    # revise memory this long after the last exchange with Buddy
MIN_GAP_SECONDS = 60  # and no more often than this, so background chatter is batched
UNSURE = 0.6          # below this, background() marks an item as a guess
GATE_MIN = 0.5        # below this, an item is left out of the gate's brief


class Memory:
    def __init__(self, client: LLMClient):
        self.client = client

        self.context: list[Event] = []
        self.history: list[ConversationTurn] = []
        self.situation = Situation()
        self.user_memory = UserMemory()
        self.goal_memory = GoalMemory()

        self._last_interaction = time.time()  # last user turn, or Buddy still busy answering
        self._last_update = time.time()
        # ids of events and turns each memory has already taken in, so a failed update retries them
        self._summarized: dict[str, set] = {"situation": set(), "user_memory": set(), "goal_memory": set()}

    def add_event(self, event: Event):
        self.context.append(event)

    def add_turn(self, turn: ConversationTurn):
        self.history.append(turn)
        self._last_interaction = time.time()

    def pending(self) -> list[UserTurn]:
        """User turns nobody has started answering yet."""
        return [t for t in self.history if isinstance(t, UserTurn) and t.response is Response.PENDING]

    # --- Updates (each blocks for one model call) ----------------------------

    def update_situation(self):
        self._update("situation", situation_updater)

    def update_user_oriented_memory(self):
        self._update("user_memory", user_updater)

    def update_goal_oriented_memory(self):
        self._update("goal_memory", goal_updater)

    def _update(self, attr: str, updater):
        seen = self._summarized[attr]
        events = [e for e in list(self.context) if e.id not in seen]
        turns = [t for t in list(self.history) if t.id not in seen and not _in_progress(t)]
        if not events and not turns:
            return

        current = getattr(self, attr)
        start = time.time()
        update = self.client.parse(
            system=updater.system_prompt(),
            conversation=notes.request(notes.describe(updater.sections(current)), events, turns),
            output=updater.Update,
        )
        new = updater.apply(current, update)
        setattr(self, attr, new)
        seen.update(x.id for x in events + turns)

        logger.info(f"{updater.LABEL} updated from {len(events)} events and {len(turns)} turns "
                    f"({time.time() - start:.1f}s):\n{notes.summary(updater.sections(new))}")
        record(attr, new)

    def watch(self, busy: Callable[[], bool] = lambda: False,
              quiet_seconds: float = QUIET_SECONDS, min_gap_seconds: float = MIN_GAP_SECONDS):
        """Run forever: update all three memories once there's something new, `quiet_seconds`
        have passed since the user and Buddy last talked (`busy` means Buddy is still answering),
        and `min_gap_seconds` since the last update. Meant for its own thread."""
        while True:
            time.sleep(1)
            now = time.time()
            if busy():
                self._last_interaction = now
                continue
            if now - self._last_interaction < quiet_seconds or now - self._last_update < min_gap_seconds \
                    or not self._has_unsummarized():
                continue
            for update in (self.update_situation, self.update_user_oriented_memory, self.update_goal_oriented_memory):
                try:
                    update()
                except Exception:
                    logger.exception(f"{update.__name__} failed")
            self._last_update = time.time()  # after a failure too, so it isn't retried every second

    def _has_unsummarized(self) -> bool:
        ids = {x.id for x in list(self.context) + list(self.history)}
        return any(ids - seen for seen in self._summarized.values())

    # --- What others read ------------------------------------------------------

    def background(self) -> str:
        """For the response model's system prompt: everything known, as context, not as topics."""
        s, u, g = self.situation, self.user_memory, self.goal_memory
        lines = []
        add = lambda label, facts: lines.append(f"{label}: " + "; ".join(_item(f) for f in facts)) if facts else None
        add("Place", [s.place] if s.place else [])
        add("Going on", [s.activity] if s.activity else [])
        add("People around", s.people)
        add("About the user", u.about)
        add("People in the user's life", u.people)
        add("The user's preferences", u.preferences)
        add("Long-term goals", [x for x in g.long_term if x.status is Status.OPEN])
        add("Current tasks", [x for x in g.tasks if x.status is Status.OPEN])
        add("Recently finished", [x for x in g.long_term + g.tasks if x.status is Status.DONE])
        if not lines:
            return ""
        return ("## What you've gathered so far\n\n"
                "Use this to understand the user and their surroundings, but don't bring it up unless it helps "
                "answer what they said. Items marked (unsure) are guesses.\n\n" + "\n".join(lines))

    def gate_brief(self) -> str | None:
        """For the gate: a few words on the present moment. Short, since the gate reads every token."""
        s = self.situation
        parts = [f.text for f in (s.place, s.activity) if f and f.confidence >= GATE_MIN]
        people = [f.text for f in s.people if f.confidence >= GATE_MIN]
        if people:
            parts.append("around: " + ", ".join(people))
        return "; ".join(parts)[:240] or None


def _item(f) -> str:
    return f.text + (" (unsure)" if f.confidence < UNSURE else "") + \
        (f", progress: {f.progress}" if getattr(f, "progress", None) else "")


def _in_progress(turn: ConversationTurn) -> bool:
    """Turns whose outcome isn't settled yet; they'll be taken in next time."""
    if isinstance(turn, UserTurn):
        return turn.response in (Response.PENDING, Response.RESPONDING)
    return turn.delivery is Delivery.IN_PROGRESS
