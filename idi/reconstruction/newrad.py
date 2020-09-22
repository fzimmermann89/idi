import math
import numpy as np
import numba, functools
import contextlib


@contextlib.contextmanager
def corrfunction(shape, z, qmax, xcenter=None, ycenter=None):
    y, x = np.meshgrid(np.arange(shape[1], dtype=np.float64), np.arange(shape[0], dtype=np.float64))
    x -= xcenter or shape[0] / 2.0
    y -= ycenter or shape[0] / 2.0
    d = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    qx, qy, qz = [(k / d * z) for k in (x, y, z)]

    def corr(input, finner):
        input = np.asarray(input).astype(np.float64, copy=False)
        return np.sum(finner(input, qx, qy, qz), axis=0)

    def inner(input, qx, qy, qz):
        out = np.zeros((shape[0] + 10, qmax), dtype=np.float64)
        for refx in numba.prange(shape[0]):
            for refy in range(shape[1]):
                refv = input[refx, refy]
                x = refx - shape[0] / 2
                y = refy - shape[1] / 2
                qstepx = abs(
                    ((z * (abs(x) + qmax / 2) / math.sqrt((abs(x) + qmax / 2) ** 2 + y ** 2 + z ** 2)))
                    - ((z * (abs(x) + (qmax / 2 - 1)) / math.sqrt((abs(x) + (qmax / 2 - 1)) ** 2 + y ** 2 + z ** 2)))
                )
                qstepy = abs(
                    ((z * (abs(y) + qmax / 2) / math.sqrt((abs(y) + qmax / 2) ** 2 + x ** 2 + z ** 2)))
                    - ((z * (abs(y) + (qmax / 2 - 1)) / math.sqrt((abs(y) + (qmax / 2 - 1)) ** 2 + x ** 2 + z ** 2)))
                )
                xmin = int(max(0, np.floor(refx - qmax / qstepx)))
                xmax = int(min(shape[0], np.ceil(refx + qmax / qstepx) + 1))
                for x in range(xmin, xmax):
                    ymin = refy
                    ymax = int(min(shape[1], 1 + np.ceil(refy + qmax / qstepy)))
                    for y in range(ymin, ymax):
                        dq = (qx[x, y] - qx[refx, refy]) ** 2 + (qy[x, y] - qy[refx, refy]) ** 2 + (qz[x, y] - qz[refx, refy]) ** 2
                        qsave = int(round(math.sqrt(dq)))
                        if qsave >= qmax:
                            break
                        val = refv * input[x, y]
                        out[refx, qsave] += val
        return out

    jitted = numba.njit(inner, parallel=True).compile("float64[:,:](float64[:,:],float64[:,:],float64[:,:],float64[:,:])")
    yield functools.partial(corr, finner=jitted)
