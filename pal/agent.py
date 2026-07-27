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
from pal.llm import Turn, get_client
from pal.types import InputObject
from pal.voice.speak import play_sound
from pal.voice.sounds import processing_tone
from pal.perception import vision_loop, understand
from pal.perception import audio_loop
from pal import interact
from pal.tools.builtin import BuiltinTools
from pal.debug.preview import preview

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

        self.spoken_words: deque = deque(maxlen=200)
        self.last_image = None

        # Controls
        self.vision_trigger = threading.Event()
        #self.start_keyboard_listener(lambda: self.vision_trigger.set())

        # Tools
        self.tool_handler = self.tool_handler_cls(self)


        print("[Start up] Initialized")

    def run(self, text_mode: bool = False):
        print("[Start up] Starting run loop")

        def feed_msg(gen):
            print("[Start up] feed_msg thread started")
            GRACE_SECONDS = 2
            for msg in gen:
                print(f"[Main loop] Message received: {msg.content if msg.content else 'empty'}")

                if self.is_speaking.is_set():
                    print("[Debug] currently speaking, ignoring input")
                    continue

                if (time.time() - self.last_spoke_at) < GRACE_SECONDS:
                    print("[Debug] within post-speech grace period, ignoring input")
                    continue
                
                # Auto-trigger camera if utterance contains visual cues
                words = set(re.findall(r"\b[\w']+\b", msg.content.lower()))
                matched = words & VISUAL_CUES
                if matched:
                    print(f"[Main loop] Visual cue detected ({matched}), triggering camera")
                    self.vision_trigger.set()

                #self.vision_trigger.set()
                print("[Main loop] Message accepted, queueing for processing")
                preview.add_event("user_speech", {"text": msg.content})
                self.internal_message_queue.put(msg)

        def process_messages():
            print("[Start up] process_messages thread started")
            while True:
                self.stop_event.clear()
                message = self.internal_message_queue.get()
                turn_start = time.time()
                query = message.content

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
                        print("[Main loop] Vision trigger pending but no image arrived")

                # Build user turn
                user_turn = Turn(role="user", text=query)
                if latest is not None:
                    crop, label, conf = latest
                    self.last_image = crop
                    with timed("encode_image"):
                        user_turn.images = [self.numpy_to_base64(crop)]
                    preview.add_event("image_attached", {"label": label, "conf": round(conf, 2)})
                    print(f"[Main loop] Image attached to message: {label} ({conf:.2f})")
                else:
                    print("[Main loop] No image queued, sending text-only")
                    #play_sound(processing_tone())
                self.conversation.append(user_turn)

                print(f"[Main loop] Processing query: {query[:50] if query else 'empty'}")
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
                    elapsed_ms = (time.time() - turn_start) * 1000
                    print(f"[Timing] time_to_first_spoken_word: {elapsed_ms:.0f}ms")
                except Exception:
                    print("[Main loop] Turn failed:")
                    traceback.print_exc(file=sys.stdout)
                finally:
                    self.is_speaking.clear()
                    self.last_spoke_at = time.time()
                print(f"[Main loop] Response complete, history length: {len(self.conversation)}")
                self.trim_conversation_history()

        input_source = self._text_input() if text_mode else self._listen()
        threading.Thread(target=feed_msg, args=(input_source,), daemon=True).start()
        threading.Thread(target=process_messages, daemon=True).start()

        def feed_img(gen):
            print("[Start up] feed_img thread started")
            for img in gen:
                crop, label, conf = img
                self.internal_image_queue.put(img)
                print(f"[Main loop] Image queued: {label} ({conf:.2f}), queue size: {self.internal_image_queue.qsize()}")


        threading.Thread(target=feed_img, args=(self._see(),), daemon=True).start()
        print("[Start up] All threads started")
        
    # Senses
    def _see(self):
        print("[Start up] _see started")
        return vision_loop.run(self.vision_trigger)
    
    def _listen(self):
        print("[Start up] _listen started")
        return audio_loop.run()
    
    def _text_input(self):
        """Replaces _listen: read messages from stdin instead of mic."""
        print("[Start up] _text_input started")
        print("[Text mode] Type messages and press Enter. Ctrl+D to quit.")
        while True:
            try:
                line = input("> ").strip()
            except EOFError:
                print("\n[Text mode] EOF, exiting")
                break
            except KeyboardInterrupt:
                print("\n[Text mode] interrupted")
                break
            if not line:
                continue
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
                    print(f"[Main loop] History trimmed, dropped {dropped} entries")
                    break

        # Strip images from older user turns (keep only on most recent)
        user_indices = [i for i, t in enumerate(self.conversation) if t.role == "user"]
        stripped = 0
        for i in user_indices[:-1]:
            if self.conversation[i].images:
                self.conversation[i].images = []
                stripped += 1
        if stripped:
            print(f"[Main loop] Stripped images from {stripped} older turns")

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
