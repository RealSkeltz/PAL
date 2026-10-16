"""Everything Buddy knows so far.

    context     every event that wasn't a conversation turn, as it arrived
    history     the conversation between the user and Buddy

The router writes events and turns in; the gate and the responder read.
"""
from pal.schemas.event import Event
from pal.schemas.turn import ConversationTurn, Response, UserTurn


class Memory:
    def __init__(self):
        self.context: list[Event] = []
        self.history: list[ConversationTurn] = []

    def add_event(self, event: Event):
        self.context.append(event)

    def add_turn(self, turn: ConversationTurn):
        self.history.append(turn)

    def pending(self) -> list[UserTurn]:
        """User turns nobody has started answering yet."""
        return [t for t in self.history if isinstance(t, UserTurn) and t.response is Response.PENDING]
