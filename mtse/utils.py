import contextlib
import time
import torch
from typing import Any, List

@contextlib.contextmanager
def time_block(name):
    try:
        duration = -time.time()
        yield
    finally:
        duration += time.time()
        print(f"{name} took {duration}sec")

def tensor2list(t: torch.Tensor | Any) -> List[Any]:
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().tolist()
    return t