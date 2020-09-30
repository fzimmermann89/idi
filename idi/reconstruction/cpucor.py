import math
import numpy as np
import numba, functools
import contextlib


@contextlib.contextmanager
def corrfunction(shape, z, qmax, xcenter=None, ycenter=None, threads=-1):
    """
    CPU based direct Autocorrelation with q correction

    parameters:
    shape (tuple) of inputs in pixels
    z (scalar) distance of detector in pixels
    optional
    xcenter (scalar): position of center in x direction, defaults to shape[0]/2
    ycenter (scalar): position of center in x direction, defaults to shape[1]/2
    threads (even int): cpu threads. -1 means use global setting. will always use even number of threads. will also use more memory!

    returns a function with signature float[:,:,:](float[:,:] image) that does the correlation
    """

    if threads == -1:
        parallel = True
        try:
            pmax = numba.get_num_threads() // 2
        except:  # old numba version
            pmax = min(8, numba.config.multiprocessing.cpu_count() // 4)
    elif threads == 1 or threads == 0:
        pmax = 1
        parallel = False
    else:
        parallel = True
        pmax = parallel // 2
    xcenter = xcenter or shape[0] / 2.0
    ycenter = ycenter or shape[1] / 2.0
    y, x = np.meshgrid(np.arange(shape[1], dtype=np.float64), np.arange(shape[0], dtype=np.float64))
    x -= xcenter
    y -= ycenter
    d = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    qx, qy, qz = [(k / d * z) for k in (x, y, z)]
    maxdqz = int(math.ceil(math.sqrt(z ** 2 - (max(qmax + 1, np.max(qx), np.max(qy)) - (qmax + 1)) ** 2)) - np.min(qz)) + 1
    # the work will be split in two blocks with different qx-offset in the result,
    # because of the curvature, they have to look a bit 'backwards' to get all pais that belong to their otput part
    overscany = int(np.ceil(np.max(np.max(qy, axis=0) - np.min(qy, axis=0))))
    overscanx = int(np.ceil(np.max(np.max(qx, axis=1) - np.min(qx, axis=1))))

    del x, y, d

    def inner(input, qx, qy, qz):
        out = np.zeros((pmax, 2 * maxdqz + 1, qmax + 1, 2 * qmax + 1), dtype=np.float64)
        for p in numba.prange(pmax):  # parallel threads working on diffrent cops of the output
            for d in numba.prange(2):  # parallel threads splitting in dqx
                direction = -1 + 2 * d  # -1 +1
                for refx in range(p, shape[0], pmax):
                    for refy in range(shape[1]):
                        qxr = qx[refx, refy]
                        qyr = qy[refx, refy]
                        qzr = qz[refx, refy]
                        refv = input[refx, refy]
                        dqx = 0
                        x = max(0, min(input.shape[0] - 1, refx - direction * overscanx))
                        while -qmax <= dqx <= qmax and 0 <= x < input.shape[0]:
                            dqy = 0
                            y = max(0, refy - overscany)
                            while dqy <= qmax and 0 <= y < input.shape[1]:
                                dqy = int(round(qy[x, y] - qyr))
                                dqx = int(round(qx[x, y] - qxr))
                                dqz = int(round(qz[x, y] - qzr))
                                val = refv * input[x, y]
                                if 0 <= dqy <= qmax and d <= direction * dqx <= qmax: #dont do dx=0 twice
                                    dqxs = dqx + qmax
                                    dqys = dqy
                                    dqzs = dqz + maxdqz
                                    out[p, dqzs, dqys, dqxs] += val
                                y += 1
                            x += direction
        return out

    finner = numba.njit(inner, parallel=parallel, fastmath=True).compile("float64[:,:,:,:](float64[:,:],float64[:,:],float64[:,:],float64[:,:])")

    def corr(input):
        """
        Do the correlation
        """
        if finner is None:
            raise ValueError("already closed, use within with statement")
        input = np.asarray(input).astype(np.float64, copy=False)
        if not all(i == s for (i, s) in zip(input.shape, shape)):
            raise ValueError("input has not the shape specified in init!")
        tmp = np.sum(finner(input, qx, qy, qz), axis=0)  # summing over the threads
        return np.concatenate((np.flip(tmp[:, :, :], axis=(0, 1, 2)), tmp[:, 1:, :]), axis=1)[:, :, :]  # unwapping

    yield corr
    qx = qy = qz = finner = shape = None
    for x in locals():
        del x
