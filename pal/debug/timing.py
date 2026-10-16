import logging
import time
from contextlib import contextmanager

logger = logging.getLogger(__name__)


@contextmanager
def timed(label: str):
    t0 = time.time()
    try:
        yield
    finally:
        elapsed_ms = (time.time() - t0) * 1000
        logger.debug(f"{label}: {elapsed_ms:.0f}ms")