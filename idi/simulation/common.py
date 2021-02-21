import numpy as _np


def rebin(arr, n):
    shape = (arr.shape[0], arr.shape[1] // (2 ** n), (2 ** n), arr.shape[2] // (2 ** n), (2 ** n))
    return arr.reshape(shape).mean(-1).mean(-2)


def randomphotons(probs, Nphotons):
    return _np.random.poisson((probs * (1 / probs.sum(axis=(1, 2)))[:, None, None]) * Nphotons)
