"""One turn of the conversation between the user and PAL.

The conversation history is the direct part of the event stream: a turn is
a Language event with role DIRECT, said by the user or by PAL. Everything
else PAL perceives stays a plain event and reaches the model as context,
not as a turn.

A PAL turn is one model response. When the model calls tools, it records
them as Actions (with their results), and the response after the tools
return is the next PAL turn. That keeps the order the model saw intact.

Unlike events, turns are not frozen: a turn has a lifecycle. A user turn
waits for a response, gets one, or has it cut off; a PAL turn is spoken
word by word until it finishes or the user barges in.

This is PAL's own record. `pal.llm.base.Turn` is the provider-neutral
wire format a client sends to the model; converting goes one way, from this
to that.
"""
from dataclasses import field
from enum import StrEnum
from typing import Annotated, Literal
from uuid import UUID, uuid4

from pydantic import Field
from pydantic.dataclasses import dataclass

from pal.schemas.event import Action, Actor, Observation, Role, Spoken, Written


class Response(StrEnum):
    PENDING = "pending"        # nobody has started answering yet
    RESPONDING = "responding"  # a PAL turn replying to this is underway
    RESPONDED = "responded"
    CANCELLED = "cancelled"    # the reply was cut off before it finished
    FAILED = "failed"          # answering raised an error


class Delivery(StrEnum):
    IN_PROGRESS = "in_progress"
    DONE = "done"
    CANCELLED = "cancelled"


def _check_utterance(utterance: Spoken | Written, actor: Actor):
    if utterance.role is not Role.DIRECT:
        raise ValueError("a turn's utterance must be direct; context events stay out of the conversation")
    if utterance.actor is not actor:
        raise ValueError(f"this turn's utterance must be said by {actor.value}")


@dataclass(kw_only=True)
class UserTurn:
    kind: Literal["user"] = "user"
    id: UUID = field(default_factory=uuid4)
    utterance: Spoken | Written
    attachments: list[Observation] = field(default_factory=list)  # e.g. the frame the user asked about
    # What automatic recall found for this question. Kept with the turn, so later replies
    # can see what this one was based on instead of finding a claim with nothing behind it.
    recalled: str = ""
    response: Response = Response.PENDING

    def __post_init__(self):
        _check_utterance(self.utterance, Actor.USER)


@dataclass(kw_only=True)
class PalTurn:
    kind: Literal["pal"] = "pal"
    id: UUID = field(default_factory=uuid4)
    in_reply_to: list[UUID]                    # the user turns this answers, often one
    utterance: Spoken | Written | None = None  # None when PAL only called tools
    actions: list[Action] = field(default_factory=list)
    delivery: Delivery = Delivery.IN_PROGRESS
    cancelled_at_word: int | None = None       # index into utterance.text.split() of the word cut off

    def __post_init__(self):
        if self.utterance is None and not self.actions:
            raise ValueError("a turn needs an utterance or actions")
        if self.utterance is not None:
            _check_utterance(self.utterance, Actor.PAL)
        if any(a.actor is not Actor.PAL or a.tool is None for a in self.actions):
            raise ValueError("a turn's actions are PAL's tool calls")
        if (self.cancelled_at_word is not None) != (self.delivery is Delivery.CANCELLED):
            raise ValueError("cancelled_at_word is set exactly when delivery is cancelled")

    def cancel(self, at_word: int):
        self.delivery = Delivery.CANCELLED
        self.cancelled_at_word = at_word

    @property
    def spoken_text(self) -> str:
        """What the user actually heard: the full text, or the words before the cut."""
        if self.utterance is None:
            return ""
        if self.cancelled_at_word is None:
            return self.utterance.text
        return " ".join(self.utterance.text.split()[:self.cancelled_at_word])


ConversationTurn = Annotated[UserTurn | PalTurn, Field(discriminator="kind")]
