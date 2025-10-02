import contextlib
import time

@contextlib.contextmanager
def time_block(name):
    try:
        duration = -time.time()
        yield
    finally:
        duration += time.time()
        print(f"{name} took {duration}sec")
