"""The capture loop: a rolling buffer of frames for looks, and optionally object recognition.

A look sends whole frames from the buffer (`camera`). With `detect`, YOLOE also runs on
every frame (`detect`) and objects are followed and recognized (`objects`); the preview
shows all of it.

Run `python -m pal.perception.vision.see` to watch recognition on the webcam.
"""
import logging
import queue
import time
from collections import deque
from typing import Generator, Iterable

import cv2
import numpy as np

from pal.debug.preview import preview
from pal.perception.vision.camera import BUFFER_SECONDS, FRAMES_PER_LOOK, TARGET_FPS, camera, downscale, pick
from pal.perception.vision.detect import detect as detect_objects
from pal.perception.vision.objects import Objects

logger = logging.getLogger(__name__)

__all__ = ["run", "TARGET_FPS"]


def run(requests: queue.Queue, detect: bool = True, frames: Iterable[np.ndarray] | None = None,
        objects: Objects | None = None) -> Generator[list[tuple[np.ndarray, str, float, float]], None, None]:
    """Capture on request and yield a list of (image, label, confidence, taken_at).

    A request is the time (epoch seconds) the question started. A look yields
    FRAMES_PER_LOOK whole frames from the buffer, spread from then to now. With
    `detect`, YOLOE also runs on every frame to follow and recognize objects
    (see `Objects`); the preview shows what it found.

    `frames` replaces the webcam, e.g. with a remote device's camera; it sets the pace.
    `objects` is the recognition state to fill, so the agent can read and change what it found.
    """
    logger.info(f"capture loop started (detector {'on' if detect else 'off'})")
    buffer = deque()  # (time, downscaled frame), oldest first
    if detect and objects is None:
        objects = Objects()

    for frame in camera() if frames is None else frames:
        now = time.time()
        buffer.append((now, downscale(frame)))
        while buffer[0][0] < now - BUFFER_SECONDS:
            buffer.popleft()

        if detect:
            r = detect_objects(frame)
            objects.update(r)
            preview.set_frame(objects.render(r))
        else:
            preview.set_frame(frame)

        try:
            since = requests.get_nowait()
        except queue.Empty:
            continue

        capture = [(image, "scene", 1.0, at) for at, image in pick(buffer, since, FRAMES_PER_LOOK)]
        ago = ", ".join(f"{now - at:.1f}s" for *_, at in capture)
        logger.info(f"looking: {len(capture)} frame(s) from {ago} ago")
        preview.show_sent([(image, now - at) for image, *_, at in capture])
        yield capture


def main():
    """Watch detection and recognition on the webcam, in a window and in the preview. q quits."""
    import webbrowser
    from pal.log import setup
    setup()
    preview.start(port=4200)
    webbrowser.open("http://127.0.0.1:4200")
    objects = Objects()
    for frame in camera():
        r = detect_objects(frame)
        objects.update(r)
        view = objects.render(r)
        preview.set_frame(view)
        cv2.imshow("pal: objects", view)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
