"""Frames: the laptop webcam, and picking frames from a rolling buffer for a look."""
from collections import deque
from typing import Generator

import cv2
import numpy as np

from pal.debug.quiet import suppress_stderr

TARGET_FPS = 10
DELAY = int(1000 / TARGET_FPS)

# The last few seconds of frames, so a look can go back to when the user started
# talking: by the time the gate has decided, what they asked about may be gone.
BUFFER_SECONDS = 6
# Frames sent per look, spread from the start of the utterance to now. One sends
# just the newest frame.
FRAMES_PER_LOOK = 3

# A whole frame is large, so cap its long edge before it becomes base64 in a prompt.
MAX_FRAME_EDGE = 768


def downscale(frame: np.ndarray) -> np.ndarray:
    """Shrink a frame so its long edge is at most MAX_FRAME_EDGE."""
    h, w = frame.shape[:2]
    scale = MAX_FRAME_EDGE / max(h, w)
    if scale >= 1:
        return frame
    return cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def camera() -> Generator[np.ndarray, None, None]:
    """BGR frames from the laptop webcam, at about TARGET_FPS."""
    # Only the camera handshake is noisy — suppressing wider than this would
    # redirect fd 2 process-wide for the life of the generator, hiding every
    # thread's tracebacks.
    with suppress_stderr():
        cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield frame
        cv2.waitKey(DELAY)


def pick(buffer: deque, since: float, count: int) -> list[tuple[float, np.ndarray]]:
    """`count` (time, frame) pairs spread from `since` to the newest frame, oldest
    first. Fewer when the buffer is too short to give distinct frames."""
    newest = buffer[-1][0]
    start = min(max(since, buffer[0][0]), newest)
    if count == 1:
        targets = [newest]
    else:
        targets = [start + (newest - start) * i / (count - 1) for i in range(count)]
    picks = []
    for target in targets:
        closest = min(buffer, key=lambda entry: abs(entry[0] - target))
        if not picks or picks[-1] is not closest:
            picks.append(closest)
    return picks
