import numpy as _np
import numba as _numba
from .common import _getidx

pmax = 16  # todo
debug = True


@_numba.njit()
def _ks(xi, yi, z, center):
    x = xi.astype(_numba.float64) - center[0]
    y = yi.astype(_numba.float64) - center[1]
    d = _np.sqrt(x ** 2 + y ** 2 + z ** 2)
    kx = x / d * z
    ky = y / d * z
    kz = z / d * z
    return kx, ky, kz


@_numba.njit()
def _qsize(Nx, Ny, z):
    qlenx = int(Nx)
    qleny = int(Ny * 2)
    qlenz = 2 * int((z - (z ** 2 / _np.sqrt((Nx / 2 + 1) ** 2 + (Ny / 2 + 1) ** 2 + z ** 2))) + 1)
    return qlenx, qleny, qlenz


@_numba.njit(parallel=False, boundscheck=not debug)
def _corr(input, z):
    Nx, Ny = input.shape
    xi, yi = _np.where(input > 0)
    Nhits = len(xi)
    qlenx, qleny, qlenz = _qsize(Nx, Ny, z)
    tmp = _np.zeros((qlenx, qleny, qlenz), dtype=_np.uint64)
    center = (Nx / 2.0, Ny / 2.0)
    kx, ky, kz = _ks(xi, yi, z, center)

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
            if 0 <= qx < qlenx and 0 <= qy < qleny and 0 <= qz < qlenz:
                tmp[qx, qy, qz] += input[xi[n], yi[n]] * input[xi[m], yi[m]]
    return tmp


@_numba.njit(parallel=True, fastmath=True, boundscheck=not debug)
def _pcorr(input, z):
    Nx, Ny = input.shape
    xi, yi = _np.where(input > 0)
    Nhits = len(xi)
    qlenx, qleny, qlenz = _qsize(Nx, Ny, z)
    tmp = [_np.zeros((qlenx, qleny, qlenz), dtype=_np.float64) for p in range(pmax)]
    center = (Nx / 2.0, Ny / 2.0)
    kx, ky, kz = _ks(xi, yi, z, center)

    for p in _numba.prange(pmax):
        ptmp = tmp[p]
        idx = _getidx(p, pmax, Nhits)
        for n in idx:
            for m in range(n):
                qz = int(round((kz[n] - kz[m])))
                qx = int(round((kx[n] - kx[m])))
                qy = int(round((ky[n] - ky[m])))
                if qx < 0:
                    qz = -qz
                    qy = -qy
                    qx = -qx
                qz = qz + qlenz // 2
                qy = qy + qleny // 2
                if 0 <= qx < qlenx and 0 <= qy < qleny and 0 <= qz < qlenz:
                    ptmp[qx, qy, qz] += input[xi[n], yi[n]] * input[xi[m], yi[m]]
    out = _np.zeros((qlenx, qleny, qlenz))

    for p in range(pmax):
        out += tmp[p]
    return out


def corr(input, z):
    """
    correlation of hits
    in NxN array or nx(NxN) arrays
    """
    if input.ndim == 2:
        return _pcorr(input, z)
    elif input.ndim == 3:
        return _np.sum(_pcorrs(input, z), axis=0)
    else:
        raise TypeError


@_numba.njit(parallel=True, boundscheck=not debug)
def _pcorrs(input, z):
    Nx, Ny = input.shape[-2:]
    Nimg = input.shape[0]
    qlenx = int(Nx)
    qleny = int(2 * Ny)
    qlenz = 2 * int((z - (z ** 2 / _np.sqrt((Nx / 2 + 1) ** 2 + (Ny / 2 + 1) ** 2 + z ** 2))) + 1)
    out = _np.zeros((pmax, qlenx, qleny, qlenz), dtype=_np.uint64)
    for p in _numba.prange(pmax):
        # Nperp = int(_np.floor(Nimg / pmax))
        # for n in range(p * Nperp, (p + 1) * Nperp):
        for n in range(p, Nimg, pmax):
            out[p] += _corr(input[n, ...], z)
    return out


def unwrap(input):
    ret = _np.zeros((2 * input.shape[0], input.shape[1], input.shape[2]))
    ret[1 : input.shape[0] + 1, 1:, 1:] = input[::-1, :0:-1, :0:-1]
    ret[input.shape[0] :, :, :] += input
    return ret


# should give a pattern after unwrapping...
# tst=np.zeros((5,10,12))
# tst[::2,:,tst.shape[2]//2]+=1
# tst[:,::2,tst.shape[2]//2]+=1
# tstu=unwrap(tst)
# plt.matshow(tstu[:,:,6],vmin=0,vmax=2)
