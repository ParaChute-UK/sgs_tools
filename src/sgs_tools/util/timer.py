import contextlib
from time import time

unit_map = {"ms": 1e-3, "s": 1, "m": 60, "min": 60, "h": 3600, "hour": 3600}


# simple timer
@contextlib.contextmanager
def timer(label, units="s", title=""):
    if title:
        print(f"{title}")
    tstart = time()
    yield
    elapsed = time() - tstart
    print(f"{label}: {elapsed / unit_map[units]:g} {units}")
