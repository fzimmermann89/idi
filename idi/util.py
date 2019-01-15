import numba as _numba
import numpy as _np


def radial_profile(data, center=None, domask=True):
    if center is None:
        center = _np.array(data.shape) // 2
    x, y = _np.indices(data.shape)
    r = _np.rint((_np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2))).astype(_np.int)
    if domask:
        r = r * (data != 0)
    rr = _np.ravel(r)
    nr = _np.bincount(rr)
    tbin = _np.bincount(rr, data.ravel())
    radialprofile = tbin / nr
    if domask:
        radialprofile[0] = 0
        radialprofile = _np.nan_to_num(radialprofile)
    return radialprofile


def cutnan(array):
    ind0 = ~_np.all(_np.isnan(array), axis=0)
    ind1 = ~_np.all(_np.isnan(array), axis=1)
    return array[ind1, :][:, ind0]


def rebin(arr, n):
    shape = (arr.shape[0] // 2 ** n, 2 ** n, arr.shape[1] // 2 ** n, 2 ** n, arr.shape[2] // 2 ** n, 2 ** n)
    return arr.reshape(shape).mean(-1).mean(1).mean(-2)


# https://stackoverflow.com/a/29042041
def bin(ndarray, new_shape, operation='sum'):
    ops = ['sum', 'mean', 'max', 'min']
    operation = operation.lower()
    if operation not in ops:
        raise ValueError("Operation not supported.")
    if ndarray.ndim != len(new_shape):
        raise ValueError("Shape mismatch: {} -> {}".format(ndarray.shape, new_shape))
    compression_pairs = [(d, c // d) for d, c in zip(new_shape, ndarray.shape)]
    flattened = [l for p in compression_pairs for l in p]
    ndarray = ndarray.reshape(flattened)
    for i in range(len(new_shape)):
        op = getattr(ndarray, operation)
        ndarray = op(-1 * (i + 1))
    return ndarray


def centered_part(arr, newshape):
    # Return the center newshape portion of the array.
    newshape = _np.asarray(newshape)
    currshape = _np.array(arr.shape)
    startind = (currshape - newshape) // 2
    endind = startind + newshape
    myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]
    return arr[tuple(myslice)]


@_numba.jit()
def find_center(img, msk, x0=0, y0=0, maxr=500, d=60):
    id = _find_center_jit(img, msk.astype(bool), int(x0), int(y0), int(maxr), int(d))
    return _np.unravel_index(id, (2 * d + 1, 2 * d + 1)) - _np.array([d, d])


@_numba.njit(parallel=True)
def _find_center_jit(img, msk, x0, y0, maxr, d):
    sx, sy = img.shape
    out = _np.empty((2 * d + 1, 2 * d + 1))
    for xs in _numba.prange(x0 - d, x0 + d + 1, 1):
        for ys in xrange(y0 - d, y0 + d + 1, 1):
            cx = sx // 2 + xs
            cy = sy // 2 + ys
            rx = min(cx, sx - cx, maxr)
            ry = min(cy, sy - cy, maxr)
            err = 0
            cn = 0
            for x in xrange(-rx + 1, 0):
                for y in xrange(-ry + 1, ry):
                    if msk[cx + x, cy + y] == 1 and msk[cx - x, cy - y] == 1:
                        cn += 1
                        err += abs(img[cx + x, cy + y] - img[cx - x, cy - y])
            out[xs + d, ys + d] = err / cn
    return out.argmin()


@_numba.vectorize([_numba.float64(_numba.complex128), _numba.float32(_numba.complex64)], target='parallel')
def abs2(x):
    return x.real * x.real + x.imag * x.imag


@_numba.vectorize([_numba.complex128(_numba.complex128), _numba.complex64(_numba.complex64)], target='parallel')
def abs2c(x):
    return x.real * x.real + x.imag * x.imag + 0j


def fill(data, invalid=None):
    import scipy.ndimage as nd

    if invalid is None:
        invalid = _np.isnan(data)
    ind = nd.distance_transform_edt(invalid, return_distances=False, return_indices=True)
    return data[tuple(ind)]


#     import scipy.ndimage
#     normnorm=cor(norm)
#     mask=normnorm<2e7
#     mask=scipy.ndimage.morphology.binary_dilation(mask,scipy.ndimage.morphology.generate_binary_structure(3,3),iterations=1)
#     _np.multiply(normnorm,(n+1),out=normnorm)
#     _np.divide(res,normnorm,out=res)

#     res[mask]=0
#     m=_np.mean(res[~mask])
#     _np.subtract(res,max(1,m),out=res)
#     res=_np.nan_to_num(res,copy=False)
#     _np.multiply(res,255,out=res)
#     res=_np.roll(res,shift=(res.shape[1]//2,res.shape[2]//2),axis=(1,2))
#     res=bin(res,_np.array(res.shape)//_np.array((1,2,2)),'max')
#     _np.clip(res,0,255,out=res);
