from abc import ABC, abstractmethod
import re
import time
import queue
from collections import deque
import threading
import cv2
import base64
from pynput import keyboard
from Shared.Classes.Message import Message
from Perception.video import vision_loop, understand
from Perception.audio import audio_loop
from Interaction import interact
from controls.headset_controls import start_headset_listener


class Pal(ABC):
    def __init__(self, system_prompt: str, model):
        # Identity
        self.system_prompt: str = system_prompt
        self.model: str = model

        # Input processing
        self.internal_message_queue: queue = queue.Queue()
        self.internal_image_queue: queue = queue.Queue()
        self.conversation: list[dict] = []

        # Output processing
        self.stop_event = threading.Event()
        self.is_speaking = threading.Event()
        self.last_spoke_at = 0.0

        self.spoken_words: deque = deque(maxlen=200)

        # Controls
        self.vision_trigger = threading.Event()
        #start_headset_listener(lambda: self.vision_trigger.set())

        self.start_keyboard_listener(lambda: self.vision_trigger.set())


        print("[Start up] Initialized")

    def run(self):
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
                
                self.vision_trigger.set()
                print("[Main loop] Message accepted, queueing for processing")
                self.internal_message_queue.put(msg)

        def process_messages():
            print("[Start up] process_messages thread started")
            while True:
                self.stop_event.clear()
                message = self.internal_message_queue.get()
                query = message.content 

                crop = None
                try:
                    img_message = self.internal_image_queue.get_nowait()
                    crop, label, conf = img_message 
                    print(f"[Main loop] Image attached to message: {label} ({conf:.2f})")

                except queue.Empty:
                    print("[Main loop] No image queued, sending text-only")
                    pass

                # Build user turn, attaching image if present
                user_msg = {"role": "user", "content": query}
                if crop is not None:
                    user_msg["images"] = [self.numpy_to_base64(crop)]
                self.conversation.append(user_msg)

                # Process the message
                print(f"[Main loop] Processing query: {query[:50] if query else 'empty'}")
                self.is_speaking.set()
                assistant_chunks = []
                for chunk in interact.run(self.conversation, self.system_prompt, self.model):
                    if self.stop_event.is_set():
                        print("[Main loop] Interrupted by user")
                        break
                    self.spoken_words.extend(re.findall(r"\b[\w']+\b", chunk.lower()))
                    assistant_chunks.append(chunk)

                full_response = "".join(assistant_chunks)
                if full_response:
                    self.conversation.append({"role": "assistant", "content": full_response})
                    print(f"[Main loop] Assistant turn saved ({len(full_response)} chars)")

                self.is_speaking.clear()
                self.last_spoke_at = time.time()
                print(f"[Main loop] Response complete, history length: {len(self.conversation)}")
                self.trim_conversation_history()

        threading.Thread(target=feed_msg, args=(self._listen(),), daemon=True).start()
        threading.Thread(target=process_messages, daemon=True).start()

        def feed_img(gen):
            print("[Start up] feed_img thread started")
            for img in gen:
                crop, label, conf = img
                self.internal_image_queue.put(img)
                print(f"[Main loop] Image queued: {label} ({conf:.2f}), queue size: {self.internal_image_queue.qsize()}")


        threading.Thread(target=feed_img, args=(self._see(),), daemon=True).start()
        #threading.Thread(target=process_images, daemon=True).start()
        print("[Start up] All threads started")
        
    # Senses
    def _see(self):
        print("[Start up] _see started")
        return vision_loop.run(self.vision_trigger)
    
    def _listen(self):
        print("[Start up] _listen started")
        return audio_loop.run()

    # Helper

    def trim_conversation_history(self):
        MAX_TURNS = 20 
        if len(self.conversation) > MAX_TURNS:
            dropped = len(self.conversation) - MAX_TURNS
            self.conversation = self.conversation[-MAX_TURNS:]
            print(f"[Main loop] History trimmed, dropped {dropped} turns")
        
        stripped = 0
        for msg in self.conversation[:-1]:
            if msg.pop("images", None) is not None:
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
