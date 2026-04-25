from abc import ABC, abstractmethod
import re
import time
import queue
from collections import deque
import threading
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
        self.trigger = threading.Event()
        start_headset_listener(lambda: self.trigger.set())

        print("[Start up] Initialized")

    def run(self):
        print("[Start up] Starting run loop")

        def feed_msg(gen):
            print("[Start up] feed_msg thread started")
            GRACE_SECONDS = 2
            for msg in gen:
                print(f"[Main loop] Message received: {msg.content if msg.content else 'empty'}")

                separate_words = re.findall(r"\b[\w']+\b", msg.content.lower())
                #print(f"[Debug] separate words: {separate_words}")
                #print(f"[Debug] spoken sentence: {self.spoken_words}")

                if self.is_speaking.is_set():
                    #print("[Debug] currently speaking, ignoring input")
                    #print("[Main loop] Potential barge-in detected")
                    continue

                if (time.time() - self.last_spoke_at) < GRACE_SECONDS:
                    print("[Debug] within post-speech grace period, ignoring input")
                    continue

                    spoken_set = set(self.spoken_words)
                    if set(separate_words) & spoken_set:
                        print("[Debug] own output recognised - no barge-in")
                        continue

                self.internal_message_queue.put(msg)

        def process_messages():
            print("[Start up] process_messages thread started")
            while True:
                self.stop_event.clear()
                message = self.internal_message_queue.get()
                query = message.content 

                # Append user turn to conversation
                self.conversation.append({"role": "user", "content": query})

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

                self.is_speaking.clear()
                self.last_spoke_at = time.time()
                print(f"[Main loop] Response complete, history length: {len(self.conversation)}")
                self.trim_conversation_history()

        threading.Thread(target=feed_msg, args=(self._listen(),), daemon=True).start()
        threading.Thread(target=process_messages, daemon=True).start()

        def feed_img(gen):
            for img in gen:
                self.internal_image_queue.put(img)
        def process_images():
            while True:
                img_message = self.internal_image_queue.get()
                r, closest_box = img_message.messages[0].content
                self.interpret_image(r, closest_box)

        #threading.Thread(target=feed_img, args=(self._see(),), daemon=True).start()
        #threading.Thread(target=process_images, daemon=True).start()
        print("[Start up] All threads started")
        
    # Senses
    def _see(self):
        print("[Start up] _see started")
        return vision_loop.run(self.trigger)
    
    def _listen(self):
        print("[Start up] _listen started")
        return audio_loop.run()

    def interpret_image(self, r, closest_box):
        if closest_box is None:
            return
        crop, label, conf = vision_loop.extract_crop(r, closest_box)
        print(f"Selected: {label} ({conf:.2f})")
        print(f"[Main loop] Interpreting: {label} ({conf:.2f})")

        def _run():
            print(f"[Main loop] Running vision model on: {label}")
            response = understand.understand_image(crop, label)
            print(f"Description: {response}")
            print(f"[Main loop] Vision response: {response[:100] if response else 'empty'}")

        threading.Thread(target=_run, daemon=True).start()

    # Helper

    def trim_conversation_history(self):
        MAX_TURNS = 20 
        if len(self.conversation) > MAX_TURNS:
            print("History trimmed")
            self.conversation = self.conversation[-MAX_TURNS:]
