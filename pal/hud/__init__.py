"""Everything drawn on top of a camera frame.

Separate from `perception` on purpose: perception decides what is out there,
the HUD decides how it looks. `frame` composes a finished image, `panel` draws
the status chrome, `typography` puts real type on the frame, and `style` holds
the tokens both of them share.
"""

from pal.hud.frame import annotate

__all__ = ["annotate"]
