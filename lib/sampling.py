from __future__ import annotations

import random
from typing import TypeVar

T = TypeVar("T")


def deterministic_sample(items: list[T], size: int, seed: int) -> list[T]:
    # Shuffle once before slicing so boundary cases consume randomness identically.
    rng = random.Random(seed)
    indices = list(range(len(items)))
    rng.shuffle(indices)
    n = min(size, len(items))
    return [items[i] for i in indices[:n]]
