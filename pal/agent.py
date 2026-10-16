"""Wires PAL together. Each stage runs on its own thread:

    senses ──► events ──► router ──► history ──► responder ──► voice
                  camera ──► captures ──┘ (attached to user turns)

The router only appends user turns and wakes the responder; the responder
answers every pending user turn at once, so speech that arrives while PAL is
talking is answered next instead of dropped. `self.cancel` cuts off the reply
in progress; barge-in will set it.
"""
import logging
import queue
import threading
import time
from abc import ABC
from uuid import uuid4

import cv2

from pal.llm import get_client
from pal.config import CAPTURES_DIR
from pal.perception.hearing import listen
from pal.perception.vision import see
from pal.response.responder import respond
from pal.routing.router import EventRouter
from pal.log import record
from pal.schemas.event import Actor, Event, Observation, Role, Written
from pal.schemas.turn import ConversationTurn, Response, UserTurn
from pal.tools.builtin import BuiltinTools
from pal.voice.sounds import capture_tone
from pal.voice.voice import Voice

logger = logging.getLogger(__name__)


def _thread(target, *args):
    threading.Thread(target=target, args=args, daemon=True).start()


class Pal(ABC):
    # Subclasses override with their own ToolHandler to add agent-specific tools.
    tool_handler_cls = BuiltinTools

    def __init__(self, system_prompt: str, model: str):
        # Identity: `model` is a "provider:name" spec, e.g. "ollama:qwen3.5:9b"
        self.system_prompt = system_prompt
        self.model = model
        self.client = get_client(model)

        # State
        self.history: list[ConversationTurn] = []
        self.context: list[Event] = []
        self.last_image = None  # read by the save_last_image tool
        self.last_spoke_at = 0.0

        # Plumbing
        self.events: queue.Queue = queue.Queue()
        self.captures: queue.Queue = queue.Queue()
        self.voice = Voice()
        self.cancel = threading.Event()

        self.idle = threading.Event()
        self.idle.set()
        self._wake = threading.Event()
        self._lock = threading.Lock()

        # Controls
        self.vision_trigger = threading.Event()
        self.detect = False  # YOLO on by default; run(detect=False) turns it off

        self.tool_handler = self.tool_handler_cls(self)
        self.router = EventRouter(
            self.history, self.context, self.captures, self.vision_trigger,
            on_user_turn=self._on_user_turn,
            last_spoke_at=lambda: self.last_spoke_at,
        )
        logger.info("initialized")

    def run(self, text_mode: bool = False, detect: bool = True):
        """Start the agent. `detect` toggles YOLO object detection on the camera."""
        self.detect = detect
        self.voice.start()
        _thread(self._feed, self._text_input() if text_mode else listen.run())
        _thread(self._feed_captures, see.run(self.vision_trigger, detect=self.detect))
        _thread(self.router.run, self.events)
        _thread(self._respond_loop)
        logger.info("all threads started")

    # --- Stages --------------------------------------------------------------

    def _feed(self, source):
        for event in source:
            record("event", event)
            self.events.put(event)

    def _feed_captures(self, captures):
        for crop, label, conf in captures:
            self.voice.play(capture_tone())
            self.last_image = crop
            capture_id = uuid4()
            path = CAPTURES_DIR / f"{capture_id}.jpg"
            cv2.imwrite(str(path), crop)
            observation = Observation(
                id=capture_id, actor=Actor.PAL, role=Role.CONTEXT,
                description=f"camera capture of {label}", label=label, confidence=conf,
                image_path=str(path),
            )
            record("capture", observation)
            self.captures.put(observation)
            logger.info(f"captured {label} ({conf:.2f})")

    def _on_user_turn(self, turn: UserTurn):
        with self._lock:
            self.idle.clear()
            self._wake.set()

    def _respond_loop(self):
        while True:
            self._wake.wait()
            self._wake.clear()
            pending = [t for t in self.history if isinstance(t, UserTurn) and t.response is Response.PENDING]
            if pending:
                self.cancel.clear()
                respond(self.history, pending, self.system_prompt, self.client, self.voice,
                        tools=self.tool_handler, cancel=self.cancel)
                self.last_spoke_at = time.time()
            with self._lock:
                if not self._wake.is_set():
                    self.idle.set()

    # --- Inputs --------------------------------------------------------------

    def _text_input(self):
        """Replaces the microphone: read messages from stdin."""
        print("[Text mode] Type messages and press Enter. Ctrl+D to quit.")
        while True:
            self.idle.wait()
            try:
                line = input("\nUser input: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n[Text mode] exiting")
                break
            if not line:
                continue
            self.idle.clear()
            yield Written(actor=Actor.USER, role=Role.DIRECT, text=line)
