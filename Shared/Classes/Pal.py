from abc import ABC, abstractmethod
import queue
import threading
from Shared.Classes.Message import Message
from Perception.video import vision_loop, understand
from Perception.audio import audio_loop
from Interaction import interact
from controls.headset_controls import start_headset_listener


class Pal(ABC):
    def __init__(self, system_prompt: str, model):
        # Identity
        self.system_prompt = system_prompt
        self.model = model

        # Input processing
        self.internal_message_queue = queue.Queue()
        self.internal_image_queue = queue.Queue()
        self.internal_history = []

        # Controls
        self.trigger = threading.Event()
        start_headset_listener(lambda: self.trigger.set())

        print("[Pal] Initialized")

    def run(self):
        print("[Pal] Starting run loop")
        def feed_img(gen):
            for img in gen:
                self.internal_image_queue.put(img)

        def feed_msg(gen):
            print("[Pal] feed_msg thread started")
            for msg in gen:
                print(f"[Pal] Message received: {msg.content[:50] if msg.content else 'empty'}")
                self.internal_message_queue.put(msg)

        def process_messages():
            print("[Pal] process_messages thread started")
            while True:
                message = self.internal_message_queue.get()
                query = message.content
                print(f"[Pal] Processing query: {query[:50] if query else 'empty'}")
                for chunk in interact.run(query, self.system_prompt, self.model):
                    self.internal_history.append(chunk)
                print(f"[Pal] Response complete, history length: {len(self.internal_history)}")

        def process_images():
            while True:
                img_message = self.internal_image_queue.get()
                r, closest_box = img_message.messages[0].content
                self.interpret_image(r, closest_box)

        #threading.Thread(target=feed_img, args=(self._see(),), daemon=True).start()
        #threading.Thread(target=process_images, daemon=True).start()

        threading.Thread(target=feed_msg, args=(self._listen(),), daemon=True).start()
        threading.Thread(target=process_messages, daemon=True).start()
        print("[Pal] All threads started")
        
    # Senses
    def _see(self):
        print("[Pal] _see started")
        return vision_loop.run(self.trigger)
    
    def _listen(self):
        print("[Pal] _listen started")
        return audio_loop.run()

    def interpret_image(self, r, closest_box):
        if closest_box is None:
            return
        crop, label, conf = vision_loop.extract_crop(r, closest_box)
        print(f"Selected: {label} ({conf:.2f})")
        print(f"[Pal] Interpreting: {label} ({conf:.2f})")

        def _run():
            print(f"[Pal] Running vision model on: {label}")
            response = understand.understand_image(crop, label)
            print(f"Description: {response}")
            print(f"[Pal] Vision response: {response[:100] if response else 'empty'}")

        threading.Thread(target=_run, daemon=True).start()
