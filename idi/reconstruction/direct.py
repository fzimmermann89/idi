#!/bin/env python
from __future__ import division as _future_div, print_function as _future_print
import numpy as _np
import numba as _numba
from six.moves import range  # for python2 compatibility
from .common import _getidx

pmax = 16  # todo


@_numba.njit(parallel=False)
def _corr(input, z):
    Nx, Ny = input.shape
    xi, yi = _np.where(input > 0)
    Nhits = len(xi)
    qlenx = int(Nx)
    qleny = int(Ny * 2)
    qlenz = 2 * int((z - (z ** 2 / _np.sqrt((Nx / 2 + 1) ** 2 + (Ny / 2 + 1) ** 2 + z ** 2))) + 1)
    tmp = _np.zeros((qlenx, qleny, qlenz), dtype=_np.uint64)
    x = (xi).astype(_numba.float64) - Nx / 2.0
    y = (yi).astype(_numba.float64) - Ny / 2.0
    d = _np.sqrt(x ** 2 + y ** 2 + z ** 2)
    kx = x / d * z
    ky = y / d * z
    kz = z / d * z

    for n in range(Nhits):
        for m in range(n):
            qz = int(_np.rint((kz[n] - kz[m])))
            qx = int(_np.rint((kx[n] - kx[m])))
            qy = int(_np.rint((ky[n] - ky[m])))
            if qx < 0:
                qz = -qz
                qy = -qy
                qx = -qx
            qz = qz + qlenz // 2
            qy = qy + qleny // 2
            # if qx >= qlenx or qy >= qleny or qz >= qlenz or qx<0 or qy<0 or qz<0:
            #    print((qx,qy,qx))
            #    pass
            # else:
            tmp[qx, qy, qz] += input[xi[n], yi[n]] * input[xi[m], yi[m]]
    return tmp


@_numba.njit(nogil=True, parallel=True, fastmath=True)
def _pcorr(input, z):
    Nx, Ny = input.shape
    xi, yi = _np.where(input > 0)
    Nhits = len(xi)
    #     qlenz = 300 #2*int(_np.ceil(z - _np.sqrt(z ** 2 - N ** 2 / 4))) #might be wrong
    qlenx = int(Nx)
    qleny = int(Ny * 2)
    qlenz = 2 * int((z - (z ** 2 / _np.sqrt((Nx / 2 + 1) ** 2 + (Ny / 2 + 1) ** 2 + z ** 2))) + 1)
    tmp = [_np.zeros((qlenx, qleny, qlenz), dtype=_np.float64) for p in range(pmax)]
    print(tmp[0].shape, len(tmp))
    x = (xi).astype(_numba.float64) - Nx / 2.0
    y = (yi).astype(_numba.float64) - Ny / 2.0
    d = _np.sqrt(x ** 2 + y ** 2 + z ** 2)
    kx = x / d * z
    ky = y / d * z
    kz = z / d * z

    for p in _numba.prange(pmax):
        ptmp = tmp[p]
        idx = _getidx(p, pmax, Nhits)
        for n in idx:
            for m in range(n):
                qz = int(_np.rint((kz[n] - kz[m])))
                qx = int(_np.rint((kx[n] - kx[m])))
                qy = int(_np.rint((ky[n] - ky[m])))
                if qx < 0:
                    qz = -qz
                    qy = -qy
                    qx = -qx
                qz = qz + qlenz // 2
                qy = qy + qleny // 2
                # if qx >= qlenx or qy >= qleny or qz >= qlenz or qx<0 or qy<0 or qz<0:
                #    print((qx,xi[n],xi[m],kx[n],kx[m]))
                # else:
                ptmp[qx, qy, qz] += input[xi[n], yi[n]] * input[xi[m], yi[m]]
    out = _np.zeros((qlenx, qleny, qlenz))

    for p in range(pmax):
        out += tmp[p]
    return out


@_numba.njit(nogil=True)
def _getidx(p, pmax, idmax):
    idsperP = idmax // (2 * pmax)
    n = 2 * idsperP
    missing = idmax - (idsperP * (2 * pmax))
    if p < missing:
        if p < missing - pmax:
            n = n + 2
            out = _np.empty(n, dtype=_numba.int64)
            out[-1] = idmax - p - pmax - 1
            out[-2] = idmax - p - 1
        else:
            n = n + 1
            out = _np.empty(n, dtype=_numba.int64)
            out[-1] = idmax - p - 1
    else:
        out = _np.empty(n, dtype=_numba.int64)
    for (m, x) in enumerate(range(p * idsperP, (p + 1) * idsperP)):
        out[m] = x
    for (k, x) in enumerate(range(((2 * pmax - 1) - p) * idsperP, ((2 * pmax) - p) * idsperP)):
        out[m + k + 1] = x
    return out


def corr(input, z):
    """
    correlation
    for one NxN array or nx(NxN) arrays
    """
    _numba.config.NUMBA_NUM_THREADS = pmax
    if input.ndim == 2:
        return _pcorr(input, z)
    elif input.ndim == 3:
        return _np.sum(_pcorrs(input, z), axis=0)
    else:
        raise TypeError


@_numba.njit(parallel=True)
def _pcorrs(input, z):
    Nx, Ny = input.shape[-2:]
    Nimg = input.shape[0]
    qlenx = int(Nx)
    qleny = int(2 * Ny)
    qlenz = 2 * int((z - (z ** 2 / _np.sqrt((Nx / 2 + 1) ** 2 + (Ny / 2 + 1) ** 2 + z ** 2))) + 1)
    Nperp = int(_np.floor(Nimg / pmax))
    out = _np.zeros((pmax, qlenx, qleny, qlenz), dtype=_np.uint64)
    for p in _numba.prange(pmax):
        for n in range(p * Nperp, (p + 1) * Nperp):
            out[p] += _corr(input[n, ...], z)
    return out


def unwrap(input):
    ret = np.zeros((2 * input.shape[0], input.shape[1], input.shape[2]))
    ret[1 : input.shape[0], 1:, 1:] = input[:0:-1, :0:-1, :0:-1]
    ret[input.shape[0] :, :, :] += input[..., :]
    return ret


# should give a pattern after unwrapping...
# tst=np.zeros((5,10,12))
# tst[::2,:,tst.shape[2]//2]+=1
# tst[:,::2,tst.shape[2]//2]+=1
# tstu=unwrap(tst)
# plt.matshow(tstu[:,:,6],vmin=0,vmax=2)
