import numba as _numba
import numpy as _np
import numexpr as _ne
from ..util import rotation as _rotation


@_numba.njit(parallel=True)
def _decaysum(a, t, tau):
    # reference without scan
    x = _np.zeros_like(a)
    for j in _numba.prange(x.shape[0]):
        x[j, 0] = a[j, 0]
        for i in range(1, x.shape[1]):
            decay = _np.exp(-(t[j, i] - t[j, i - 1]) / tau)
            x[j, i] = a[j, i] + decay * x[j, i - 1]
    return x


@_numba.njit(parallel=True)
def _psort(a, b):
    '''
    idx = np.argsort(a, axis=-1)
    a = a[np.arange(a.shape[0])[:, None], idx]
    b = b[np.arange(a.shape[0])[:, None], idx]
    '''
    for i in _numba.prange(len(a)):
        ids = _np.argsort(a[i])
        a[i, :] = a[i][ids]
        b[i, :] = b[i][ids]


def _ab2(x):
    return x.imag ** 2 + x.real ** 2


def _integral(amp, t0, tau):
    """
    integrates the exponential decay of |x|^2 with supports t0 and decay tau
    """
    t0s = _np.atleast_2d(t0)
    amps = _np.atleast_2d(amp)
    _psort(t0s, amps)
    i = _ab2(_decaysum(amps, t0s, tau))
    td = _np.diff(t0s, axis=-1)
    return _np.sum(-tau / 2 * i[:, :-1] * _np.expm1(-2 * td / tau), axis=-1) + tau / 2 * i[:, -1]


def simulate(simobject, Ndet, pixelsize, detz, k, c, tau, pulsewidth, detangles=(0, 0), settings='', threads=None, *args, **kwargs):
    """
    Time dependent simulation with decaying amplitudes (cpu version with few optimisations).
    simobject: simobject to use for simulation (in lengthunit)
    pixelsize: pixelsize (in lengthunit)
    detz: Detector-sample distance
    k: angular wave number (in 1/lengthunit)
    c: speed of light in (lengthunit/timeunit)
    tau: decay time (in timeunit)
    pulsewidth: FWHM of gaussian exciation pulse (in timeunit)
    settings: string, can contain
        scale - do 1/r intensity scaling
    threads: blocksize to do multiple pixels at once (memory tradeoff)
    """

    from math import sin, cos

    if _np.size(Ndet) == 1:
        Ndet = [Ndet, Ndet]
    if 'scale' in settings:
        eq = 'exp(-1j*(s*k-phases))/d'
    else:
        eq = 'exp(-1j*(s*k-phases))'
    n = _np.prod(Ndet)
    if threads is None:
        threads = 4
    # do highest power of two of n <= threads pixels at once. tradeoff between memory allocations and call overhead.
    blocksize = min(threads, (n & (~(n - 1))))

    dets = _np.array(
        _np.meshgrid(pixelsize * (_np.arange(Ndet[0]) - (Ndet[0] / 2)), pixelsize * (_np.arange(Ndet[1]) - (Ndet[1] / 2)), detz)
    ).T.reshape(-1, 3)
    if _np.any(detangles):
        dets = dets @ _rotation(*detangles, 0).T

    res = _np.zeros(Ndet[0] * Ndet[1]).reshape(-1, blocksize)
    data = simobject.get()
    times = _np.random.randn(simobject.N) * (pulsewidth / 2.35)

    # this could be parallelised but array sharing might be annoying to implement, so either mpi run everything or be happy with the threaded parts
    for j, det in enumerate(dets.reshape(-1, blocksize, 3)):
        d = _np.linalg.norm(det, axis=-1)[:, None]
        q = det / d
        q[..., -1] -= 1  # incoming beam along z direction
        phases = data[:, 3]
        s = _np.inner(q, data[:, :3])  # path difference
        e = _ne.evaluate(eq)  # complex e field
        t = -s / c + times.T  # arrival time
        res[j] = _integral(e, t, tau)
    res = res.reshape(Ndet[0], Ndet[1])
    return res


def simulate_gen(
    simobject, Ndet, pixelsize, detz, k, c, tau, pulsewidth, detangles=(0, 0), settings='', maximg=_np.inf, threads=None, *args, **kwargs
):
    """
    Time dependent simulation with decaying amplitudes (cpu version with few optimisations).
    simobject: simobject to use for simulation (in lengthunit)
    pixelsize: pixelsize (in lengthunit)
    detz: Detector-sample distance
    k: angular wave number (in 1/lengthunit)
    c: speed of light in (lengthunit/timeunit)
    tau: decay time (in timeunit)
    pulsewidth: FWHM of gaussian exciation pulse (in timeunit)
    settings: string, can contain
        scale - do 1/r intensity scaling
    threads: blocksize to do multiple pixels at once (memory tradeoff)
    """

    # as each simulation is slow compared to setup, we dont currently do any work ahead and just call simulate()
    i = 0
    while i < maximg:
        yield simulate(simobject, Ndet, pixelsize, detz, k, c, tau, pulsewidth, detangles, settings, threads, *args, **kwargs)
        i += 1
