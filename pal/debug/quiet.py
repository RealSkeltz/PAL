"""Keep known, harmless library noise out of the terminal."""
import contextlib
import os
import warnings


@contextlib.contextmanager
def suppress_stderr():
    """Mute file descriptor 2, which also catches output from C libraries."""
    with open(os.devnull, "w") as devnull:
        old_stderr = os.dup(2)
        os.dup2(devnull.fileno(), 2)
        try:
            yield
        finally:
            os.dup2(old_stderr, 2)
            os.close(old_stderr)


def quiet_startup():
    """Call first thing, before anything imports cv2 or torch."""
    # OpenCV and PyAV (a faster-whisper dependency) each bundle FFmpeg's
    # libavdevice, and the macOS runtime warns about the duplicate classes when
    # the second one loads. Load both now with stderr muted; later imports
    # reuse them and stay silent.
    with suppress_stderr():
        import av  # noqa: F401
        import cv2  # noqa: F401

    # silero-vad still loads its model with torch.jit.load.
    warnings.filterwarnings("ignore", message=r"`torch\.jit\.load` is deprecated", category=FutureWarning)
