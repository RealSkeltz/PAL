"""Sort incoming events into conversation and context.

Language from the user that is meant for PAL becomes a UserTurn in the
history; everything else is context. Typed input is direct by nature. Speech
is direct when it names PAL or comes right after PAL spoke; only when neither
holds does the slower model gate decide.
"""
import queue
import re
import threading
from dataclasses import replace
from typing import Callable

from pal.debug.preview import preview
from pal.routing.gate import addressed_by_name, should_speak, within_conversation_window
from pal.log import logger
from pal.routing.visual_cues import VISUAL_CUES
from pal.schemas.event import Actor, Event, Language, Observation, Role
from pal.schemas.turn import ConversationTurn, UserTurn

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
        logger.info("[Start up] router started")
        while True:
            self.route(events.get())

    def route(self, event: Event):
        if not isinstance(event, Language):
            self.context.append(event)
            return

        logger.info(f"[Heard]    {event.actor.value}: {event.text!r}")
        preview.add_event("user_speech", {"text": event.text})

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
            logger.info("[Gate]     addressed by name -> speak")
            return True
        if within_conversation_window(self.last_spoke_at()):
            logger.info("[Gate]     within conversation window -> speak")
            return True
        return self._gate(event)

    def _gate(self, event: Language) -> bool:
        try:
            decision = should_speak(event.text, None, self.history)
        except Exception:
            logger.exception("gate failed, responding anyway")
            return True
        logger.info(f"[Gate]     {decision.reason} ({decision.probability:.2f}) -> {'speak' if decision.speak else 'silent'}")
        return decision.speak

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
                logger.info("[Image]    vision trigger pending but no image arrived")
        return [latest] if latest is not None else []

    def _drain_captures(self) -> Observation | None:
        latest = None
        while True:
            try:
                latest = self.captures.get_nowait()
            except queue.Empty:
                return latest
            self.context.append(latest)
