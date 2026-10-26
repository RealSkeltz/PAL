"""Sort incoming events into conversation and context.

Language from the user that is meant for PAL becomes a UserTurn in the
history; everything else is context. Typed input is direct by nature. Speech
is direct when it names PAL or comes right after PAL spoke; only when neither
holds does the model gate decide. The same gate call decides whether the
answer needs the camera, and only then are frames attached: from when the
user started talking up to now, since that's what they were looking at.
"""
import logging
import queue
from dataclasses import replace
from typing import Callable

from pal.routing.gate import addressed_by_name, assess, within_conversation_window
from pal.log import record
from pal.memory import Memory
from pal.schemas.event import Actor, Event, Language, Observation, Role, Spoken
from pal.schemas.turn import UserTurn

logger = logging.getLogger(__name__)

CAPTURE_WAIT_SECONDS = 1.0
# Silence after speech before an utterance counts as finished (listen.COMMIT_AFTER;
# not imported, since importing listen loads the speech models).
UTTERANCE_TAIL_SECONDS = 0.5


class EventRouter:
    def __init__(
        self,
        memory: Memory,
        captures: queue.Queue,
        look_requests: queue.Queue,
        on_user_turn: Callable[[UserTurn], None],
        last_spoke_at: Callable[[], float],
    ):
        self.memory = memory
        self.captures = captures
        self.look_requests = look_requests
        self.on_user_turn = on_user_turn
        self.last_spoke_at = last_spoke_at

    def run(self, events: queue.Queue):
        logger.info("started")
        while True:
            self.route(events.get())

    def route(self, event: Event):
        if not isinstance(event, Language):
            self.memory.add_event(event)
            return

        logger.info(f"heard {event.actor.value}: {event.text!r}", extra={"speech": (event.actor.value, event.text)})

        if event.actor is not Actor.USER:
            self.memory.add_event(event)
            return

        speak, look = self._assess(event)
        if not speak:
            self.memory.add_event(event)
            return

        event = replace(event, role=Role.DIRECT)
        turn = UserTurn(utterance=event, attachments=self._attachments(look, _started_at(event)))
        self.memory.add_turn(turn)
        self.on_user_turn(turn)

    def _assess(self, event: Language) -> tuple[bool, bool]:
        """(is it for PAL, does the answer need the camera), logged and recorded."""
        if event.role is Role.DIRECT:
            via = "typed"
        elif addressed_by_name(event.text):
            via = "name"
        elif within_conversation_window(self.last_spoke_at()):
            via = "window"
        else:
            via = "model"

        try:
            speak_decision, look_decision = assess(event.text, self.memory.history, ask_speak=via == "model",
                                                       situation=self.memory.gate_brief())
        except Exception:
            logger.exception("gate failed, responding anyway, without the camera")
            speak_decision = look_decision = None
            via = "gate_error" if via == "model" else via

        speak = speak_decision.yes if speak_decision else True
        look = look_decision.yes if look_decision else False
        speak_detail = f" {speak_decision.reason} ({speak_decision.probability:.2f})" if speak_decision else ""
        look_detail = f" ({look_decision.probability:.2f})" if look_decision else ""
        logger.info(f"gate: {via}{speak_detail} -> {'speak' if speak else 'silent'}"
                    + (f", camera: {'look' if look else 'no'}{look_detail}" if speak else ""))
        record("gate", {
            "event_id": str(event.id), "via": via, "speak": speak,
            "reason": speak_decision.reason if speak_decision else None,
            "probability": speak_decision.probability if speak_decision else None,
            "look": look, "look_probability": look_decision.probability if look_decision else None,
        })
        return speak, look

    def _attachments(self, look: bool, since: float) -> list[Observation]:
        """Camera frames from `since` to now when the gate says the answer needs them."""
        self._drain_captures()  # older captures are context, not this question's subject
        if not look:
            return []
        self.look_requests.put(since)
        try:
            frames = self.captures.get(timeout=CAPTURE_WAIT_SECONDS)
        except queue.Empty:
            logger.info("camera triggered but no image arrived")
            return []
        for frame in frames:
            self.memory.add_event(frame)
        return frames

    def _drain_captures(self):
        while True:
            try:
                frames = self.captures.get_nowait()
            except queue.Empty:
                return
            for frame in frames:
                self.memory.add_event(frame)


def _started_at(event: Language) -> float:
    """When the user started saying this, in epoch seconds. Events are made once the
    utterance is finished; typed text has no duration, so it started when it arrived."""
    started = event.at.timestamp()
    if isinstance(event, Spoken) and event.duration_s:
        started -= event.duration_s + UTTERANCE_TAIL_SECONDS
    return started
