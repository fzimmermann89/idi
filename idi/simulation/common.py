import numpy as _np
import numba as _numba


def randomphotons(probs, Nphotons, rng=None, dtype=float):
    if rng is None:
        rng = _np.random.default_rng(_np.random.randint(2 ** 63))
    return rng.poisson(probs * (Nphotons / probs.sum(axis=(-2, -1), keepdims=True))).astype(dtype)


def chargesharing(img, s):
    """
    simulates charge charing between neighboring pixels:
    for each count in the discrete image img, a uniformly random position within the pixel is chosen
    and a gaussian distribution with sigma s applied to the neighboring pixels.
    """

    @_numba.njit(fastmath=True, cache=True)
    def _put(ret, hits, s, w):
        mu = _np.random.rand(len(hits), 2) - 0.5
        v = _np.zeros((2 * w + 1) * (2 * w + 1))
        for i in range(len(hits)):
            hitx, hity = hits[i]
            mux, muy = mu[i]
            j = 0
            for ox in range(-w, w + 1):
                for oy in range(-w, w + 1):
                    v[j] = _np.exp(-((ox - mux) ** 2 + (oy - muy) ** 2) / (2 * s ** 2))
                    j += 1
            v = v / _np.sum(v)
            j = 0
            for ox in range(-w, w + 1):
                for oy in range(-w, w + 1):
                    ret[hitx + ox, hity + oy] += v[j]
                    j += 1

    w = int(_np.ceil(2 * s))
    ret = _np.zeros((img.shape[0] + 2 * w, img.shape[1] + 2 * w))
    hits = _np.repeat(_np.indices(img.shape).reshape(2, -1), img.ravel(), axis=1).T
    _put(ret, hits, s, w)
    ret = ret[w:-w, w:-w]
    return ret
