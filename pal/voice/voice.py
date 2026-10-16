"""PAL's voice: speech and sounds on one ordered, stoppable output.

`say` and `play` return immediately. A synthesis thread turns text into audio
one item at a time, and the sound card pulls from the result through a
callback, so the next sentence is synthesized while the current one plays.

Because the voice owns playback, it is the one place that knows how much was
heard. Every item carries a tag; `heard(tag)` is how many of that tag's words
have played, estimated within a sentence by how far into its audio playback
got. `stop()` cuts off immediately and freezes those counts, which is what a
barge-in needs to record where PAL was interrupted.
"""
import logging
import queue
import threading
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Hashable

import numpy as np
import sounddevice as sd

from pal.debug.timing import timed
from pal.voice.tts import kokoro

logger = logging.getLogger(__name__)

SAMPLE_RATE = 24000  # Kokoro outputs 24kHz mono float32


@dataclass
class _Clip:
    samples: np.ndarray
    tag: Hashable
    words: int
    pos: int = 0


class Voice:
    def __init__(self, voice: str = "af_bella", speed: float = 1.2):
        self.voice = voice
        self.speed = speed

        self._inbox: queue.Queue = queue.Queue()  # (epoch, text or samples, tag), in order
        self._clips: deque[_Clip] = deque()       # synthesized, waiting to play
        self._current: _Clip | None = None        # playing now
        self._heard: defaultdict[Hashable, int] = defaultdict(int)
        self._pending = 0                         # items accepted but not finished playing
        self._epoch = 0                           # bumped by stop(); older items are dropped
        self._lock = threading.Lock()
        self._idle = threading.Event()
        self._idle.set()

        self._stream = sd.OutputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="float32",
            blocksize=1024, callback=self._fill,
        )

    def start(self):
        threading.Thread(target=self._synthesize, daemon=True).start()
        self._stream.start()

    # --- Public ------------------------------------------------------------

    def say(self, text: str, tag: Hashable = None):
        self._enqueue(text, tag)

    def play(self, samples: np.ndarray, tag: Hashable = None):
        """Play a sound in line with speech, e.g. a tone between two sentences."""
        self._enqueue(samples, tag)

    def stop(self):
        """Cut off now and drop everything queued."""
        with self._lock:
            if self._current is not None:
                self._heard[self._current.tag] += self._partial_words(self._current)
                self._current = None
            self._clips.clear()
            self._epoch += 1
            self._pending = 0
            self._idle.set()

    def heard(self, tag: Hashable) -> int:
        with self._lock:
            n = self._heard[tag]
            if self._current is not None and self._current.tag == tag:
                n += self._partial_words(self._current)
            return n

    @property
    def busy(self) -> bool:
        return not self._idle.is_set()

    def wait(self, cancel: threading.Event | None = None) -> bool:
        """Block until everything queued has played. False if `cancel` was set first."""
        while not self._idle.wait(timeout=0.05):
            if cancel is not None and cancel.is_set():
                return False
        return True

    # --- Internals ---------------------------------------------------------

    def _enqueue(self, item, tag):
        with self._lock:
            self._pending += 1
            self._idle.clear()
            epoch = self._epoch
        self._inbox.put((epoch, item, tag))

    def _synthesize(self):
        while True:
            epoch, item, tag = self._inbox.get()
            if epoch != self._epoch:
                continue
            try:
                if isinstance(item, str):
                    with timed(f"tts ({len(item)} chars)"):
                        samples, _ = kokoro.create(item, voice=self.voice, speed=self.speed)
                    words = len(item.split())
                else:
                    samples, words = item, 0
                clip = _Clip(np.asarray(samples, dtype=np.float32).reshape(-1), tag, words)
            except Exception:
                logger.exception("synthesis failed")
                clip = None

            with self._lock:
                if epoch != self._epoch:
                    continue  # stopped while synthesizing
                if clip is None or len(clip.samples) == 0:
                    self._finish_one()
                else:
                    self._clips.append(clip)

    def _fill(self, outdata, frames, time, status):
        """Sound card callback: copy the next `frames` samples into `outdata`."""
        out = outdata[:, 0]
        filled = 0
        with self._lock:
            while filled < frames:
                if self._current is None:
                    if not self._clips:
                        break
                    self._current = self._clips.popleft()
                clip = self._current
                n = min(frames - filled, len(clip.samples) - clip.pos)
                out[filled:filled + n] = clip.samples[clip.pos:clip.pos + n]
                clip.pos += n
                filled += n
                if clip.pos >= len(clip.samples):
                    self._heard[clip.tag] += clip.words
                    self._current = None
                    self._finish_one()
        out[filled:] = 0

    def _finish_one(self):
        self._pending -= 1
        if self._pending <= 0:
            self._pending = 0
            self._idle.set()

    @staticmethod
    def _partial_words(clip: _Clip) -> int:
        return int(clip.words * clip.pos / len(clip.samples))
