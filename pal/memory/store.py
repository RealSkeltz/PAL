"""Everything Buddy knows so far, from raw to distilled.

    context     every event that wasn't a conversation turn, as it arrived
    history     the conversation between the user and Buddy
    situation   the distilled picture of what is going on around the user

The router writes events and turns in; the gate and the responder read. The
situation is revised in the background (`watch`) once the user and Buddy
haven't talked for a while, so it never competes with a reply. Other people
talking doesn't count as talking to Buddy; their remarks are simply batched
into the next update.
"""
import logging
import time
from typing import Callable

from pal.llm.base import LLMClient
from pal.log import record
from pal.memory import situation as situation_update
from pal.schemas.event import Event
from pal.schemas.situation import Situation
from pal.schemas.turn import ConversationTurn, Delivery, Response, UserTurn

logger = logging.getLogger(__name__)

QUIET_SECONDS = 30    # revise the situation this long after the last exchange with Buddy
MIN_GAP_SECONDS = 60  # and no more often than this, so background chatter is batched


class Memory:
    def __init__(self, client: LLMClient):
        self.client = client

        self.context: list[Event] = []
        self.history: list[ConversationTurn] = []
        self.situation = Situation()

        self._last_interaction = time.time()  # last user turn, or Buddy still busy answering
        self._last_update = time.time()
        self._summarized: set = set()  # ids of events and turns already in the situation

    def add_event(self, event: Event):
        self.context.append(event)

    def add_turn(self, turn: ConversationTurn):
        self.history.append(turn)
        self._last_interaction = time.time()

    def pending(self) -> list[UserTurn]:
        """User turns nobody has started answering yet."""
        return [t for t in self.history if isinstance(t, UserTurn) and t.response is Response.PENDING]

    def update_situation(self):
        """Revise the situation with everything since the last update. Blocks for one model call."""
        events = [e for e in list(self.context) if e.id not in self._summarized]
        turns = [t for t in list(self.history) if t.id not in self._summarized and not _in_progress(t)]
        if not events and not turns:
            return

        start = time.time()
        update = self.client.parse(
            system=situation_update.system_prompt(),
            conversation=situation_update.request(self.situation, events, turns),
            output=situation_update.SituationUpdate,
        )
        self.situation = situation_update.apply(self.situation, update)
        self._summarized.update(x.id for x in events + turns)

        logger.info(f"situation updated from {len(events)} events and {len(turns)} turns "
                    f"({time.time() - start:.1f}s):\n{situation_update.summary(self.situation)}")
        record("situation", self.situation)

    def watch(self, busy: Callable[[], bool] = lambda: False,
              quiet_seconds: float = QUIET_SECONDS, min_gap_seconds: float = MIN_GAP_SECONDS):
        """Run forever: update the situation once there's something new, `quiet_seconds` have
        passed since the user and Buddy last talked (`busy` means Buddy is still answering),
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
            try:
                self.update_situation()
            except Exception:
                logger.exception("situation update failed")
            self._last_update = time.time()  # after a failure too, so it isn't retried every second

    def _has_unsummarized(self) -> bool:
        return any(e.id not in self._summarized for e in self.context) or \
            any(t.id not in self._summarized for t in self.history)


def _in_progress(turn: ConversationTurn) -> bool:
    """Turns whose outcome isn't settled yet; they'll be summarized next time."""
    if isinstance(turn, UserTurn):
        return turn.response in (Response.PENDING, Response.RESPONDING)
    return turn.delivery is Delivery.IN_PROGRESS
