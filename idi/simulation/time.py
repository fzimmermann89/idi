import numba
from numba import cuda
import math
import cupy as cp
import numpy as np


@numba.njit(parallel=True)
def _decaysum(a, t, tau):
    x = np.zeros_like(a)
    for j in numba.prange(x.shape[0]):
        x[j, 0] = a[j, 0]
        for i in range(1, x.shape[1]):
            decay = np.exp(-(t[j, i] - t[j, i - 1]) / tau)
            x[j, i] = a[j, i] + decay * x[j, i - 1]
    return x


def _ab2(x):
    return x.imag ** 2 + x.real ** 2


def _integral(amp, t0, tau):
    """
    integrates the exponential decay of |x|^2 with supports t0 and decay tau
    """

    idx = np.argsort(t0, axis=-1)
    t0s = np.atleast_2d(t0)
    t0s = t0s[np.arange(t0s.shape[0])[:, None], idx]
    amps = np.atleast_2d(amp)
    amps = amps[np.arange(amps.shape[0])[:, None], idx]
    i = ab2(decaysum(amps, t0s, tau))
    td = np.diff(t0s, axis=-1)
    return np.sum(-tau / 2 * i[:, :-1] * np.expm1(-2 * td / tau), axis=-1) + tau / 2 * i[:, -1]


def simulate(simobject, Ndet, pixelsize, detz, k, c, tau, verbose=True):
    if _np.size(Ndet) == 1:
        Ndet = [Ndet, Ndet]
    blocksize = 4
    dets = np.array(
        np.meshgrid(
            pixelsize * np.arange(Ndet[0]) - (Ndet[0] / 2), pixelsize * (np.arange(Ndet[1]) - Ndet[1] / 2), detz
        )
    ).T
    res = []
    g = simobject.get_time()
    for det in dets.reshape(-1, blocksize, 3):
        d = np.linalg.norm((g[:, :3] - det[:, None, :]), axis=-1)  # distance
        e = 1 / d * np.exp(1j * k * d)  # complex e field
        d = c * d + g[:, -1]  # arrival time
        d -= t.min(axis=-1)[:, None]
        i = _integral(e, d, tau)
        res.append(i)
    res = np.array(res).reshape(Ndet[0], Ndet[1])
    return res
