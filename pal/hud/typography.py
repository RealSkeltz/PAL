"""Real type on the frame, via Pillow.

OpenCV can only draw Hershey — 1960s single-stroke paths with no true
letterforms, no kerning and no weights. Nothing built on it looks finished, so
text goes through FreeType instead: SF Pro for labels, SF Mono for numbers so
digits are tabular and the latency readout stops shifting as values change.

Shapes stay in OpenCV, which anti-aliases them (`LINE_AA`) where Pillow does not.
Pillow is used only for glyphs, where FreeType's hinted grayscale AA is exactly
what is wanted.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Tried in order; the first that loads wins. SF Pro and SF Mono are variable
# fonts, so a named instance selects the weight.
FONT_STACK = {
    "sans": (
        ("/System/Library/Fonts/SFNS.ttf", None),
        ("/System/Library/Fonts/HelveticaNeue.ttc", 0),
        ("/System/Library/Fonts/Supplemental/Arial.ttf", None),
    ),
    "mono": (
        ("/System/Library/Fonts/SFNSMono.ttf", None),
        ("/System/Library/Fonts/Menlo.ttc", 0),
        ("/System/Library/Fonts/Monaco.ttf", None),
    ),
}

_cache: dict = {}


def font(family: str, size: int, weight: str = "Regular"):
    """A cached font. Falls back through the stack, then to Pillow's default."""
    key = (family, size, weight)
    if key in _cache:
        return _cache[key]

    loaded = None
    for path, index in FONT_STACK.get(family, ()):
        try:
            loaded = (ImageFont.truetype(path, size) if index is None
                      else ImageFont.truetype(path, size, index=index))
        except OSError:
            continue
        try:
            loaded.set_variation_by_name(weight)
        except (OSError, AttributeError, ValueError):
            pass  # static font, or no such named instance — regular is fine
        break

    if loaded is None:
        loaded = ImageFont.load_default()

    _cache[key] = loaded
    return loaded


def measure(text: str, face, tracking: float = 0.0) -> int:
    """Advance width of `text`, including letterspacing between glyphs."""
    if not text:
        return 0
    width = sum(face.getlength(ch) for ch in text)
    return round(width + tracking * (len(text) - 1))


def cap_height(face) -> int:
    """Height of capitals — the right thing to centre against, unlike ascent."""
    top, bottom = face.getbbox("H")[1], face.getbbox("H")[3]
    return bottom - top


class TextLayer:
    """Collects text for one frame, then composites it in a single pass.

    Converting between numpy and Pillow per string would be far more expensive
    than drawing everything onto one transparent layer and blending once.
    """

    def __init__(self, shape):
        height, width = shape[:2]
        self._size = (width, height)
        self._image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        self._draw = ImageDraw.Draw(self._image)
        self._rects: list[tuple[int, int, int, int]] = []

    def text(self, xy, string, face, color, tracking: float = 0.0, alpha: int = 255,
             halo=None):
        """Draw `string` with its left edge on the text baseline at `xy`.

        `halo` draws the same string offset one pixel in each direction first.
        A single drop shadow only protects two edges of a glyph; over video the
        background can be bright on any side.
        """
        if not string:
            return
        if halo is not None:
            x, y = xy
            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                self._render((x + dx, y + dy), string, face, halo, tracking, alpha)
        self._render(xy, string, face, color, tracking, alpha)
        self._mark(xy, string, face, tracking)

    def _mark(self, xy, string, face, tracking):
        """Record roughly where this string landed, for a tight composite later."""
        x, y = xy
        ascent, descent = face.getmetrics()
        width = measure(string, face, tracking)
        self._rects.append((int(x) - 2, int(y) - ascent - 2,
                            int(x) + width + 2, int(y) + descent + 2))

    def _render(self, xy, string, face, color, tracking, alpha):
        fill = (*color, alpha)
        if not tracking:
            self._draw.text(xy, string, font=face, fill=fill, anchor="ls")
            return
        x, y = xy
        for char in string:
            self._draw.text((x, y), char, font=face, fill=fill, anchor="ls")
            x += face.getlength(char) + tracking

    def composite(self, frame, alpha: float = 1.0):
        """Blend the collected text onto a BGR frame, in place.

        Blends each cluster of text separately. A single bounding box over
        everything would span from the status plate to a caption near the bottom
        of the frame, putting most of the image through float maths for the sake
        of a few hundred lit pixels.
        """
        if alpha <= 0:
            return
        width, height = self._size
        for x1, y1, x2, y2 in _merge(self._rects):
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)
            if x2 <= x1 or y2 <= y1:
                continue

            patch = np.asarray(self._image.crop((x1, y1, x2, y2)), dtype=np.float32)
            weight = (patch[:, :, 3:4] / 255.0) * alpha
            rgb = patch[:, :, 2::-1]                   # RGB -> BGR

            region = frame[y1:y2, x1:x2].astype(np.float32)
            frame[y1:y2, x1:x2] = (region * (1 - weight) + rgb * weight).astype(np.uint8)


def _merge(rects):
    """Collapse overlapping rects so no pixel is composited twice."""
    merged = []
    for rect in rects:
        x1, y1, x2, y2 = rect
        hit = True
        while hit:
            hit = False
            for other in list(merged):
                ox1, oy1, ox2, oy2 = other
                if x1 < ox2 and ox1 < x2 and y1 < oy2 and oy1 < y2:
                    x1, y1 = min(x1, ox1), min(y1, oy1)
                    x2, y2 = max(x2, ox2), max(y2, oy2)
                    merged.remove(other)
                    hit = True
        merged.append((x1, y1, x2, y2))
    return merged
