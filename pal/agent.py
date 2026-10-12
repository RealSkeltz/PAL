from abc import ABC, abstractmethod
import re
import sys
import time
import queue
import traceback
from collections import deque
import threading
import cv2
import base64
from pynput import keyboard

from pal.utils.timing import timed
from pal.constants.visual_cues import VISUAL_CUES
from pal.clients import Turn, get_client
from pal.types import InputObject
from pal.voice.speak import play_sound
from pal.voice.sounds import processing_tone
from pal.perception import vision_loop, understand
from pal.perception import audio_loop
from pal import interact
from pal.tools.builtin import BuiltinTools
from pal.debug.preview import preview
from pal.system_one.should_respond import should_speak
from pal.system_one.utils import heard


class Pal(ABC):
    # Subclasses override with their own ToolHandler to add agent-specific tools.
    tool_handler_cls = BuiltinTools

    def __init__(self, system_prompt: str, model: str):
        # Identity — `model` is a "provider:name" spec, e.g. "ollama:qwen3.5:9b"
        self.system_prompt: str = system_prompt
        self.model: str = model
        self.client = get_client(model)

        # Input processing
        self.internal_message_queue: queue = queue.Queue()
        self.internal_image_queue: queue = queue.Queue()
        self.conversation: list[Turn] = []

        # Output processing
        self.stop_event = threading.Event()
        self.is_speaking = threading.Event()
        self.last_spoke_at = 0.0
        # Set when the current input has been fully handled; text mode waits on it
        # so the prompt only appears once PAL's output for the turn is done.
        self.turn_finished = threading.Event()
        self.turn_finished.set()

        self.spoken_words: deque = deque(maxlen=200)
        self.last_image = None

        # Controls
        self.vision_trigger = threading.Event()
        self.detect = True   # YOLO on by default; run(detect=False) turns it off
        #self.start_keyboard_listener(lambda: self.vision_trigger.set())

        # Tools
        self.tool_handler = self.tool_handler_cls(self)


        print("[Start up] Initialized")

    def run(self, text_mode: bool = False, detect: bool = True):
        """Start the agent. `detect` toggles YOLO object detection on the camera."""
        print("[Start up] Starting run loop")
        self.detect = detect

        def feed_msg(gen):
            print("[Start up] feed_msg thread started")
            for msg in gen:
                # TODO: support barge-in. With a headset PAL can't hear itself, so this
                # no longer guards against echo; it only stops the user from interrupting
                # ("PAL, stop"). Instead, cut off TTS and start a new turn.
                if self.is_speaking.is_set():
                    print(f"[Ignored]  {msg.content!r} (PAL is speaking)")
                    self.turn_finished.set()
                    continue

                # Auto-trigger camera if utterance contains visual cues
                words = set(re.findall(r"\b[\w']+\b", msg.content.lower()))
                matched = words & VISUAL_CUES
                if matched:
                    print(f"[Vision]   visual cue {sorted(matched)}, triggering camera")
                    self.vision_trigger.set()

                #self.vision_trigger.set()
                preview.add_event("user_speech", {"text": msg.content})
                self.internal_message_queue.put(msg)

        def process_messages():
            print("[Start up] process_messages thread started")
            # Within this window after PAL spoke, the next utterance counts as a follow-up.
            # Kept short: after it, the gate sees `recent` and catches most follow-ups itself.
            CONVERSATION_WINDOW_SECONDS = 5

            turn_counter = 0
            while True:
                self.stop_event.clear()
                message = self.internal_message_queue.get()
                turn_start = time.time()
                query = message.content

                turn_counter += 1
                print(f"\n{'-' * 24} Turn {turn_counter} {'-' * 24}")
                print(f"[Heard]    {query!r}")

                # Gate before anything is queued, latched, or added to history.
                # If the gate itself fails, respond anyway rather than go quiet.
                # TODO: speech PAL stays silent on is currently dropped, but the assistant
                # may need to know what went on around it. Once world events are kept
                # separately from conversation history, record silent utterances there
                # instead of discarding them.
                decision = None
                if re.search(r"\bpal\b", query, re.IGNORECASE):
                    print("[Gate]     skipped: addressed by name")
                elif (time.time() - self.last_spoke_at) < CONVERSATION_WINDOW_SECONDS:
                    print("[Gate]     skipped: within conversation window")
                else:
                    try:
                        recent = [f"{t.role}: {t.text}" for t in self.conversation[-4:] if t.text]
                        decision = should_speak(heard(query, scene=None, recent=recent))
                        print(f"[Gate]     {'speak' if decision.speak else 'silent'} "
                              f"({decision.reason}, p={decision.probability:.2f})")
                    except Exception:
                        print("[Gate]     failed, responding anyway:")
                        traceback.print_exc(file=sys.stdout)

                if decision is not None and not decision.speak:
                    self.turn_finished.set()
                    continue

                # Drain image queue, take latest
                latest = None
                try:
                    while True:
                        latest = self.internal_image_queue.get_nowait()
                except queue.Empty:
                    pass

                # Wait briefly for image if vision trigger is pending
                if latest is None and self.vision_trigger.is_set():
                    try:
                        latest = self.internal_image_queue.get(timeout=1.0)
                    except queue.Empty:
                        print("[Image]    vision trigger pending but no image arrived")

                # Build user turn
                user_turn = Turn(role="user", text=query)
                if latest is not None:
                    crop, label, conf = latest
                    self.last_image = crop
                    with timed("encode_image"):
                        user_turn.images = [self.numpy_to_base64(crop)]
                    preview.add_event("image_attached", {"label": label, "conf": round(conf, 2)})
                    print(f"[Image]    attached: {label} ({conf:.2f})")
                else:
                    print("[Image]    none, text only")
                    #play_sound(processing_tone())
                self.conversation.append(user_turn)

                self.is_speaking.set()

                # Run the agentic turn — interact.run handles tool calls and
                # appends to self.conversation itself. A failed turn must not
                # kill this thread or leave is_speaking latched, which would
                # silently drop every subsequent utterance.
                try:
                    interact.run(
                        self.conversation,
                        self.system_prompt,
                        client=self.client,
                        tool_handler=self.tool_handler,
                    )
                except Exception:
                    print("[Error]    turn failed:")
                    traceback.print_exc(file=sys.stdout)
                finally:
                    self.is_speaking.clear()
                    self.last_spoke_at = time.time()
                elapsed_ms = (time.time() - turn_start) * 1000
                print(f"[Done]     {elapsed_ms:.0f}ms, history length {len(self.conversation)}")
                self.trim_conversation_history()
                self.turn_finished.set()

        input_source = self._text_input() if text_mode else self._listen()
        threading.Thread(target=feed_msg, args=(input_source,), daemon=True).start()
        threading.Thread(target=process_messages, daemon=True).start()

        def feed_img(gen):
            print("[Start up] feed_img thread started")
            for img in gen:
                crop, label, conf = img
                self.internal_image_queue.put(img)
                print(f"[Vision]   queued {label} ({conf:.2f}), queue size {self.internal_image_queue.qsize()}")


        threading.Thread(target=feed_img, args=(self._see(),), daemon=True).start()
        print("[Start up] All threads started")
        
    # Senses
    def _see(self):
        print("[Start up] _see started")
        return vision_loop.run(self.vision_trigger, detect=self.detect)
    
    def _listen(self):
        print("[Start up] _listen started")
        return audio_loop.run()
    
    def _text_input(self):
        """Replaces _listen: read messages from stdin instead of mic."""
        print("[Start up] _text_input started")
        print("[Text mode] Type messages and press Enter. Ctrl+D to quit.")
        while True:
            self.turn_finished.wait()
            try:
                line = input("\nUser input: ").strip()
            except EOFError:
                print("\n[Text mode] EOF, exiting")
                break
            except KeyboardInterrupt:
                print("\n[Text mode] interrupted")
                break
            if not line:
                continue
            self.turn_finished.clear()
            # Match the Message wrapper your audio_loop yields
            yield InputObject(type="text", content=line)

    # Helper
    def trim_conversation_history(self):
        MAX_ENTRIES = 40  # rough budget; tool turns add multiple entries
        if len(self.conversation) > MAX_ENTRIES:
            # Find a user message to start from to avoid orphaning tool chains
            excess = len(self.conversation) - MAX_ENTRIES
            for i in range(excess, len(self.conversation)):
                if self.conversation[i].role == "user":
                    dropped = i
                    self.conversation = self.conversation[i:]
                    print(f"[History]  trimmed, dropped {dropped} entries")
                    break

        # Strip images from older user turns (keep only on most recent)
        user_indices = [i for i, t in enumerate(self.conversation) if t.role == "user"]
        stripped = 0
        for i in user_indices[:-1]:
            if self.conversation[i].images:
                self.conversation[i].images = []
                stripped += 1
        if stripped:
            print(f"[History]  stripped images from {stripped} older turns")

    def numpy_to_base64(self, img):
        _, buffer = cv2.imencode(".jpg", img)
        return base64.b64encode(buffer).decode("utf-8")
    
  
    def start_keyboard_listener(self, callback):
        pressed = set()
        
        def on_press(key):
            if key == keyboard.Key.space:
                print("[Trigger] keyboard trigger fired")
                callback()
        
        def on_release(key):
            pressed.discard(key)
        
        listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        listener.daemon = True
        listener.start()
