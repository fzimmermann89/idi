import numpy as _np


def randomphotons(probs, Nphotons, rng=None, dtype=float):
    if rng is None:
        rng = _np.random.default_rng(_np.random.randint(2 ** 63))
    return rng.poisson(probs * (Nphotons / probs.sum(axis=(-2, -1), keepdims=True))).astype(dtype)
