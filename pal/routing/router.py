"""Sort incoming events into conversation and context.

Language from the user that is meant for PAL becomes a UserTurn in the
history; everything else is context. Typed input is direct by nature. Speech
is direct when it names PAL or comes right after PAL spoke; only when neither
holds does the slower model gate decide.
"""
import logging
import queue
import re
import threading
from dataclasses import replace
from typing import Callable

from pal.routing.gate import addressed_by_name, should_speak, within_conversation_window
from pal.log import record
from pal.routing.visual_cues import VISUAL_CUES
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
    ):
        self.history = history
        self.context = context
        self.captures = captures
        self.vision_trigger = vision_trigger
        self.on_user_turn = on_user_turn
        self.last_spoke_at = last_spoke_at

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
            return
        if event.role is not Role.DIRECT:
            if not self._is_for_pal(event):
                self.context.append(event)
                return
            event = replace(event, role=Role.DIRECT)

        turn = UserTurn(utterance=event, attachments=self._attachments(event.text))
        self.history.append(turn)
        self.on_user_turn(turn)

    def _is_for_pal(self, event: Language) -> bool:
        if addressed_by_name(event.text):
            return self._decided(event, "name", speak=True)
        if within_conversation_window(self.last_spoke_at()):
            return self._decided(event, "window", speak=True)
        try:
            decision = should_speak(event.text, None, self.history)
        except Exception:
            logger.exception("gate failed, responding anyway")
            return self._decided(event, "gate_error", speak=True)
        return self._decided(event, "model", decision.speak, decision.reason, decision.probability)

    def _decided(self, event: Language, via: str, speak: bool, reason: str | None = None,
                 probability: float | None = None) -> bool:
        detail = f" {reason} ({probability:.2f})" if reason is not None else ""
        logger.info(f"gate: {via}{detail} -> {'speak' if speak else 'silent'}")
        record("gate", {"event_id": str(event.id), "via": via, "speak": speak,
                        "reason": reason, "probability": probability})
        return speak

    def _attachments(self, text: str) -> list[Observation]:
        """The latest camera capture; if there is none and the user refers to
        something visible, take one now."""
        latest = self._drain_captures()
        if latest is None and set(re.findall(r"\b[\w']+\b", text.lower())) & VISUAL_CUES:
            self.vision_trigger.set()
            try:
                latest = self.captures.get(timeout=CAPTURE_WAIT_SECONDS)
                self.context.append(latest)
            except queue.Empty:
                logger.info("vision trigger pending but no image arrived")
        return [latest] if latest is not None else []

    def _drain_captures(self) -> Observation | None:
        latest = None
        while True:
            try:
                latest = self.captures.get_nowait()
            except queue.Empty:
                return latest
            self.context.append(latest)
