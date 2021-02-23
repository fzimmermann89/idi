import numpy as _np

def randomphotons(probs, Nphotons):
    return _np.random.poisson((probs * (1 / probs.sum(axis=(1, 2)))[:, None, None]) * Nphotons)
