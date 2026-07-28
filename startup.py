import os
import re
import sys
import threading
from contextlib import contextmanager

# Lives outside the pal package on purpose: importing pal pulls in cv2, which is
# one half of the warning this module exists to hide.
#
# cv2 and av each bundle their own libavdevice, so whichever loads second makes
# the ObjC runtime complain about duplicate AVFoundation capture classes. Neither
# library actually uses ffmpeg's avfoundation device, so the warning is noise —
# but it is written straight to fd 2 by dyld, out of reach of Python's filters.
_NOISE = re.compile(r"^objc\[\d+\]: Class AVF\w+ is implemented in both ")


def _split_line(buf: bytes) -> int:
    """Index just past the first line break, or -1 if the buffer holds no full line."""
    breaks = [i for i in (buf.find(b"\n"), buf.find(b"\r")) if i != -1]
    return min(breaks) + 1 if breaks else -1


@contextmanager
def filtered_stderr():
    """Drop known-benign native warnings while passing the rest of fd 2 through."""
    saved = os.dup(2)
    read_fd, write_fd = os.pipe()

    def pump():
        buf = b""
        while True:
            chunk = os.read(read_fd, 4096)
            if not chunk:
                break
            buf += chunk
            while (cut := _split_line(buf)) != -1:
                line, buf = buf[:cut], buf[cut:]
                if not _NOISE.match(line.decode("utf-8", "replace")):
                    os.write(saved, line)
        if buf and not _NOISE.match(buf.decode("utf-8", "replace")):
            os.write(saved, buf)
        os.close(read_fd)

    thread = threading.Thread(target=pump, daemon=True)
    thread.start()
    os.dup2(write_fd, 2)
    os.close(write_fd)
    try:
        yield
    finally:
        sys.stderr.flush()
        os.dup2(saved, 2)  # closes the pipe's last writer, so the pump sees EOF
        thread.join(timeout=1)
        os.close(saved)
