import numpy as _np
import numba as _numba
from math import erf, sqrt

try:
    import cupy as _cp
except ImportError:
    _cp = None


def randomphotons(probs, Nphotons, rng=None, dtype=float):
    if not _np.isfinite(Nphotons):
        return probs
    if isinstance(probs, _np.ndarray):
        if rng is None:
            rng = _np.random.default_rng(_np.random.randint(2**63))
        return rng.poisson(probs * (Nphotons / probs.sum(axis=(-2, -1), keepdims=True))).astype(dtype)
    elif _cp is not None and isinstance(probs, _cp.ndarray):
        if rng is None:
            rng = _cp.random.default_rng(_np.random.randint(2**63))
        elif isinstance(rng, _np.random.Generator):
            rng = _cp.random.Generator(rng.integers(high=2**63))
        return rng.poisson(probs * (Nphotons / probs.sum(axis=(-2, -1), keepdims=True))).astype(dtype)


def chargesharing(img, s, ns=0):
    """
    simulates charge sharing between neighboring pixels:
    for each count in the discrete image img, a uniformly random position within the pixel is chosen
    and a Gaussian distribution with sigma s applied to the neighboring pixels.
    charge that leaves the detector is lost.
    """

    if not s > 0:
        if ns == 0:
            return img
        else:
            return img * (1 + ns * _np.random.randn(*img.shape))

    w = int(_np.ceil(2 * s))
    hits = _np.repeat(_np.indices(img.shape).reshape(2, -1), img.astype(int).ravel(), axis=1).T + _np.array((w, w))
    ret = _np.zeros((img.shape[0] + 2 * w, img.shape[1] + 2 * w))
    __putcharges(ret, hits, s, w, ns)
    ret = ret[w:-w, w:-w]
    return ret


@_numba.njit(fastmath=True, cache=True)
def __putcharges(ret, hits, s, w, ns):
    mu = _np.random.rand(len(hits), 2) - 0.5
    k = _np.zeros((2 * w + 1, 2 * w + 1))
    c = 1 / (2 * sqrt(2) * s)
    if ns > 0:
        scale = 1 + _np.random.randn(len(hits)) * ns
    for i in range(len(hits)):
        k[:] = 1 / 4
        mux, muy = mu[i]
        for x in range(-w, w + 1):
            a = (mux - x) / (sqrt(2) * s)
            k[x + w, :] *= erf(c + a) + erf(c - a)
        for y in range(-w, w + 1):
            b = (muy - y) / (sqrt(2) * s)
            k[:, y + w] *= erf(c + b) + erf(c - b)
        hitx, hity = hits[i]
        for x in range(2 * w + 1):
            for y in range(2 * w + 1):
                ret[hitx + y - w, hity + x - w] += scale[i] * k[x, y] if ns > 0 else k[x, y]
