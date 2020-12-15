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


@cuda.jit(fastmath=True)
def _cudecaysum(x, t, tau):
    thread = cuda.threadIdx.x
    threads = cuda.blockDim.x
    b = cuda.blockIdx.x
    N = x.shape[1]
    step = 1
    if b >= x.shape[0]:
        return
    while step < N:
        for k in range(thread * 2 * step, N, 2 * step * threads):
            decay = math.exp(-(t[b, k + 2 * step - 1] - t[b, k + step - 1]) / tau)
            x[b, k + 2 * step - 1] += decay * x[b, k + step - 1]
        step = step * 2
        cuda.syncthreads()
    step = N // 2
    while step > 1:
        step = step // 2
        for k in range(thread * 2 * step, N - 2 * step, 2 * step * threads):
            decay = math.exp(-(t[b, k + 3 * step - 1] - t[b, k + 2 * step - 1]) / tau)
            x[b, 3 * step - 1 + k] += decay * x[b, 2 * step - 1 + k]
        cuda.syncthreads()


@cp.fuse()
def _cuab2(x):
    return cp.real(x) ** 2 + cp.imag(x) ** 2


@cp.fuse()
def _cuint(x, t0, t1):
    """
    actual integral, separate function to allow fusing
    to be called with arguments x[:, :-1], t[:, 1:], t[:, :-1]
    """
    return cp.sum(-tau / 2 * (_cuab2(x)) * cp.expm1(-(t0 - t1) * (2 / tau)), axis=-1)


def _cuintegrate(x, t, tau):
    """
    integrates the exponential decay of |x|^2 with supports t and decay tau
    will clobber inputs to preserve memory. N has to be power of 2!
    """
    idx = cp.argsort(t, axis=1)
    t[:] = t[np.arange(t.shape[0])[:, None], idx]
    x[:] = x[np.arange(x.shape[0])[:, None], idx]
    _cudecaysum[x.shape[0], 256](cuda.as_cuda_array(x), cuda.as_cuda_array(t), tau)
    integral = _cuint(x[:, :-1], t[:, 1:], t[:, :-1], x[:, -1])
    integral += tau / 2 * _cuab2(x[:, -1])
    return integral


def simulate(simobject, Ndet, pixelsize, detz, k, c, tau, verbose=True): 
    if _np.size(Ndet) == 1:
        Ndet = [Ndet, Ndet]
    blocksize = 4
    dets = cp.array(
        np.meshgrid(
            pixelsize * np.arange(Ndet[0]) - (Ndet[0] / 2), pixelsize * (np.arange(Ndet[1]) - Ndet[1] / 2), detz
        )
    ).T
    res = []
    g = cp.array(simobject.get_time())
    for det in dets.reshape(-1, blocksize, 3):
        d = cp.linalg.norm((g[:, :3] - det[:, None, :]), axis=-1)  # distance
        e = 1 / d * cp.exp((1j * k) * d)  # complex e field
        d = c * d + g[:, -1]  # arrival time
        d -= t.min(axis=-1)[:, None]
        i = _integral(e, d, tau)
        res.append(i)
    res = np.array(res).reshape(Ndet[0], Ndet[1])
    return res
