"""Grouped k-fold over the labelled plates: photos with the same plate number always share a fold (no twin leaks)."""

import numpy as np


def folds(plates: dict, k: int = 5, seed: int = 0) -> list[list[str]]:
    """k lists of file names, balanced in size, deterministic for a given seed."""
    groups: dict[str, list[str]] = {}
    for f, p in plates.items():
        groups.setdefault(p["digits"], []).append(f)
    names = sorted(groups)
    np.random.default_rng(seed).shuffle(names)
    names.sort(key=lambda g: -len(groups[g]))  # big groups first (stable, so ties keep the shuffled order)
    out: list[list[str]] = [[] for _ in range(k)]
    for g in names:
        min(out, key=len).extend(groups[g])
    return [sorted(o) for o in out]
