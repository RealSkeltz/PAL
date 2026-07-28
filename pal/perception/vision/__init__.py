"""Camera capture and everything drawn on top of it.

`loop` owns the camera and the detect-then-track cycle, `tracking` turns each
frame's raw detections into stable tracks, and `hud` renders those tracks onto
the frame. Callers only need `run`.
"""

from pal.perception.vision.loop import downscale, run

__all__ = ["run", "downscale"]
