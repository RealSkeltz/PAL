import time
from contextlib import contextmanager


@contextmanager
def timed(label: str):
    t0 = time.time()
    try:
        yield
    finally:
        elapsed_ms = (time.time() - t0) * 1000
        print(f"[Timing] {label}: {elapsed_ms:.0f}ms")