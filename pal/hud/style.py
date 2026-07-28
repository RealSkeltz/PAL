"""Design tokens for the on-frame HUD.

Colours live here in RGB because that is how they are written everywhere else —
CSS, design tools, the preview page. OpenCV wants BGR, so call `bgr()` at the
point of use rather than keeping a second, silently-reversed palette.

Sizes are given for a reference frame height and scaled per frame, so the HUD
holds its proportions from a 480p webcam to 1080p capture.
"""

# --- palette (RGB) ---

ACCENT = (125, 211, 192)        # #7dd3c0
PLATE = (14, 15, 17)            # #0e0f11, the preview page's background
RULE = (255, 255, 255)          # used at low alpha
TEXT = (230, 231, 234)
TEXT_DIM = (138, 141, 147)
TEXT_FAINT = (86, 90, 97)

STATE_IDLE = (140, 143, 149)
STATE_LISTENING = (125, 211, 192)   # #7dd3c0 — as "Heard" in the event log
STATE_THINKING = (183, 148, 212)    # #b794d4
STATE_SPEAKING = (201, 169, 110)    # #c9a96e — as "Said"
TOOL = (183, 148, 212)              # #b794d4 — as "Tool"

# --- alpha ---

PLATE_ALPHA = 0.78
RULE_ALPHA = 0.18
BRACKET_ALPHA = 0.55

# --- spacing, on a 4px grid ---

UNIT = 4
MARGIN = UNIT * 4               # plate to frame edge
PAD_X = UNIT * 4                # plate interior
PAD_Y = UNIT * 3
ROW_GAP = UNIT * 2
RULE_GAP = UNIT * 2
COLUMN_GAP = UNIT * 7           # minimum label-to-value gutter

# --- type ---

REFERENCE_HEIGHT = 720          # sizes below are quoted at this frame height
SIZE_HEADER = 15
SIZE_LABEL = 11
SIZE_VALUE = 13
SIZE_UNIT = 10
SIZE_CAPTION = 17

MIN_SIZE = 9
TRACKING = 0.14                 # letterspacing for caps, as a share of size


def bgr(rgb):
    return rgb[2], rgb[1], rgb[0]


def scaled(size, frame_height):
    """Scale a reference size to this frame, never below legibility."""
    return max(MIN_SIZE, round(size * frame_height / REFERENCE_HEIGHT))


def tracking_for(size):
    return size * TRACKING
