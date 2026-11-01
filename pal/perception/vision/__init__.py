"""Vision: the camera and looks (`see`, `camera`), and object recognition (`detect`, `embed`,
`stored`, `objects`, `draw`)."""
import os

# Before any of these modules imports OpenCV, which otherwise logs camera chatter on its own
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"
