"""Sort incoming events into conversation and context.

Language from the user that is meant for PAL becomes a UserTurn in the
history; everything else is context. Typed input is direct by nature. Speech
is direct when it names PAL or comes right after PAL spoke; only when neither
holds does the model gate decide. The same gate call decides whether the
answer needs the camera, and only then is a frame captured and attached.
"""
import logging
import queue
import threading
from dataclasses import replace
from typing import Callable

from pal.routing.gate import addressed_by_name, assess, within_conversation_window
from pal.log import record
from pal.schemas.event import Actor, Event, Language, Observation, Role
from pal.schemas.turn import ConversationTurn, UserTurn

logger = logging.getLogger(__name__)

CAPTURE_WAIT_SECONDS = 1.0


class EventRouter:
    def __init__(
        self,
        history: list[ConversationTurn],
        context: list[Event],
        captures: queue.Queue,
        vision_trigger: threading.Event,
        on_user_turn: Callable[[UserTurn], None],
        last_spoke_at: Callable[[], float],
        on_not_for_pal: Callable[[Language], None] = lambda event: None,
    ):
        self.history = history
        self.context = context
        self.captures = captures
        self.vision_trigger = vision_trigger
        self.on_user_turn = on_user_turn
        self.last_spoke_at = last_spoke_at
        self.on_not_for_pal = on_not_for_pal

    def run(self, events: queue.Queue):
        logger.info("started")
        while True:
            self.route(events.get())

    def route(self, event: Event):
        if not isinstance(event, Language):
            self.context.append(event)
            return

        logger.info(f"heard {event.actor.value}: {event.text!r}")

        if event.actor is not Actor.USER:
            self.context.append(event)
            self.on_not_for_pal(event)
            return

        speak, look = self._assess(event)
        if not speak:
            self.context.append(event)
            self.on_not_for_pal(event)
            return

        event = replace(event, role=Role.DIRECT)
        turn = UserTurn(utterance=event, attachments=self._attachments(look))
        self.history.append(turn)
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
            speak_decision, look_decision = assess(event.text, self.history, ask_speak=via == "model")
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

    def _attachments(self, look: bool) -> list[Observation]:
        """A fresh camera frame when the gate says the answer needs one."""
        self._drain_captures()  # older captures are context, not this question's subject
        if not look:
            return []
        self.vision_trigger.set()
        try:
            capture = self.captures.get(timeout=CAPTURE_WAIT_SECONDS)
        except queue.Empty:
            logger.info("camera triggered but no image arrived")
            return []
        self.context.append(capture)
        return [capture]

    def _drain_captures(self) -> Observation | None:
        latest = None
        while True:
            try:
                latest = self.captures.get_nowait()
            except queue.Empty:
                return latest
            self.context.append(latest)
