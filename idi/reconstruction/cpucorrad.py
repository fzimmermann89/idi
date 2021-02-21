import math
import numpy as _np
import numba as _numba
import contextlib


@contextlib.contextmanager
def corrfunction(shape, z, qmax, xcenter=None, ycenter=None):
    """
    CPU based radial Autocorrelation with q correction

    parameters:
    shape (tuple) of inputs in pixels
    z (scalar) distance of detector in pixels
    qmax (scalar): maximum distance
    optional
    xcenter (scalar): position of center in x direction, defaults to shape[0]/2
    ycenter (scalar): position of center in x direction, defaults to shape[1]/2
    returns a function with signature float[:](float[:,:] image) that does the correlation
    """
    xcenter = xcenter or shape[0] / 2.0
    ycenter = ycenter or shape[1] / 2.0
    y, x = _np.meshgrid(_np.arange(shape[1], dtype=_np.float64), _np.arange(shape[0], dtype=_np.float64))
    x -= xcenter
    y -= ycenter
    d = _np.sqrt(x ** 2 + y ** 2 + z ** 2)
    qx, qy, qz = [(k / d * z) for k in (x, y, z)]
    del x, y, d

    def inner(input, qx, qy, qz):
        out = _np.zeros((shape[0] + 10, qmax), dtype=_np.float64)
        for refx in _numba.prange(shape[0]):
            for refy in range(shape[1]):
                qxr = qx[refx, refy]
                qyr = qy[refx, refy]
                qzr = qz[refx, refy]
                refv = input[refx, refy]
                for direction in range(2):
                    dqx = 0
                    x = refx + direction  # dont dx=0 it twice
                    while -qmax <= dqx <= qmax and 0 <= x < input.shape[0]:
                        dqy = 0
                        y = refy
                        while dqy <= qmax and 0 <= y < input.shape[1]:
                            dq = (qx[x, y] - qxr) ** 2 + (qy[x, y] - qyr) ** 2 + (qz[x, y] - qzr) ** 2
                            qsave = int(round(math.sqrt(dq)))
                            if qsave >= qmax:
                                break
                            val = refv * input[x, y]
                            out[refx, qsave] += val
                            y += 1
                        x += -1 + 2 * direction
        return out

    finner = _numba.njit(inner, parallel=True, fastmath=True).compile("float64[:,:](float64[:,:],float64[:,:],float64[:,:],float64[:,:])")

    def corr(input):
        """
        Do the correlation
        """
        if finner is None:
            raise ValueError("already closed, use within with statement")
        input = _np.asarray(input).astype(_np.float64, copy=False)
        if not all(i == s for (i, s) in zip(input.shape, shape)):
            raise ValueError("not the same shape")
        return _np.sum(finner(input, qx, qy, qz), axis=0)

    yield corr
    qx = qy = qz = finner = shape = None
    for x in locals():
        del x
