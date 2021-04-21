from .accum import *
from .filters import *
from .funchelper import *
from .poissondisk import *
from .random import *
from . import h5util
import numba as _numba
import numpy as _np
import scipy.ndimage as _snd
import numexpr as _ne
import itertools as _it


def radial_profile(data, center=None, calcStd=False, os=1):
    """
    calculates a ND radial profile of data around center. will ignore nans
    calStd: calculate standard deviation, return tuple of (profile, std)
    os: oversample by a factor. With default 1 the stepsize will be 1 pixel, with 2 it will be .5 pixels etc.
    """
    if center is None:
        center = _np.array(data.shape) // 2
    if len(center) != data.ndim:
        raise TypeError("center should be of length data.ndim")
    center = _np.array(center)[tuple([slice(len(center))] + data.ndim * [None])]
    ind = _np.indices(data.shape)
    r = (_np.rint(os * _np.sqrt(((ind - center) ** 2).sum(axis=0)))).astype(int)
    databin = _np.bincount(r.ravel(), (_np.nan_to_num(data)).ravel())
    nr = _np.bincount(r.ravel(), ((~_np.isnan(data)).astype(float)).ravel())
    radialprofile = databin / nr
    if not calcStd:
        return radialprofile

    data2bin = _np.bincount(r.ravel(), (_np.nan_to_num(data ** 2)).ravel())
    radial2profile = data2bin / nr
    std = _np.sqrt(radial2profile - radialprofile ** 2)
    return radialprofile, std


def cutnan(array):
    """
    remove full-nan rows and columns of 2d array
    """
    ind0 = ~_np.all(_np.isnan(array), axis=0)
    ind1 = ~_np.all(_np.isnan(array), axis=1)
    return array[ind1, :][:, ind0]


# https://stackoverflow.com/a/29042041
def bin(ndarray, new_shape, operation="sum"):
    """
    bin an ndarray
    ndarray: nd-array
    new_shape: shape to bin to. shape of ndarray has to be integer multiple of new_shape along each dimension
    operation: string. sum, mean, max, or min. operation to use
    """
    ops = ["sum", "mean", "max", "min"]
    operation = operation.lower()
    if operation not in ops:
        raise ValueError("Operation not supported.")
    if ndarray.ndim != len(new_shape):
        raise ValueError("Shape mismatch: {} -> {}".format(ndarray.shape, new_shape))
    compression_pairs = [(d, c // d) for d, c in zip(new_shape, ndarray.shape)]
    flattened = [i for p in compression_pairs for i in p]
    ndarray = ndarray.reshape(flattened)
    for i in range(len(new_shape)):
        op = getattr(ndarray, operation)
        ndarray = op(-1 * (i + 1))
    return ndarray


def rebin(ndarray, n, operation="mean"):
    """
    rebin an ndarray
    parameters:
    ndarray: nd-array
    n: scalar or list, factor to bin with. if scalar, same factor along all dimensions
    operation: string. sum, mean, max, or min. operation to use
    if n doesnt divide arr along one dimensions, last elements of n will silently be dropped
    """
    if not (_np.size(n) == 1 or _np.size(n) == _np.ndim(ndarray)):
        raise ValueError("n should be scalar or of same length as ndarray has dimensions")
    newshape = _np.maximum(_np.array(ndarray.shape) // n, 1)
    return bin(_np.copy(ndarray[tuple((slice(None, s) for s in n * newshape))]), newshape, operation,)


def centered_part(arr, newshape):
    """
    Return the center newshape portion of the array.
    """
    newshape = _np.asarray(newshape)
    currshape = _np.array(arr.shape)
    startind = (currshape - newshape) // 2
    endind = startind + newshape
    myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]
    return arr[tuple(myslice)]


def list2array(li):
    def _len(e):
        return 1 if _np.isscalar(e) else len(e)

    maxlen = _np.max([_len(e) for e in li])
    return _np.array([_np.pad(e, (0, maxlen - _len(e)), 'constant') for e in li])


@_numba.jit()
def find_center(img, mask, x0=0, y0=0, maxr=500, d=60):
    id = _find_center_jit(img, mask.astype(bool), int(x0), int(y0), int(maxr), int(d))
    return _np.unravel_index(id, (2 * d + 1, 2 * d + 1)) - _np.array([d, d])


@_numba.njit(parallel=True)
def _find_center_jit(img, msk, x0, y0, maxr, d):
    sx, sy = img.shape
    out = _np.empty((2 * d + 1, 2 * d + 1))
    for xs in _numba.prange(x0 - d, x0 + d + 1, 1):
        for ys in range(y0 - d, y0 + d + 1, 1):
            cx = sx // 2 + xs
            cy = sy // 2 + ys
            rx = min(cx, sx - cx, maxr)
            ry = min(cy, sy - cy, maxr)
            err = 0
            cn = 0
            for x in range(-rx + 1, 0):
                for y in range(-ry + 1, ry):
                    if msk[cx + x, cy + y] == 1 and msk[cx - x, cy - y] == 1:
                        cn += 1
                        err += abs(img[cx + x, cy + y] - img[cx - x, cy - y])
            out[xs + d, ys + d] = err / cn
    return out.argmin()


@_numba.vectorize(
    [_numba.float64(_numba.complex128), _numba.float32(_numba.complex64)], target="parallel",
)
def abs2(x):
    return x.real * x.real + x.imag * x.imag


@_numba.vectorize(
    [_numba.complex128(_numba.complex128), _numba.complex64(_numba.complex64)], target="parallel",
)
def abs2c(x):
    return x.real * x.real + x.imag * x.imag + 0j


def fill(data, invalid=None):
    """
    fill invalid values by closest valid value. invalid: mask of invalid values, default: np.isnan(data)
    """
    if invalid is None:
        invalid = _np.isnan(data)
    ind = _snd.distance_transform_edt(invalid, return_distances=False, return_indices=True)
    return data[tuple(ind)]


def photons_localmax(img, E, thres=0.0):
    """
    photonize image.
    First count whole photons.
    Second count fractional/split photons at local maxima if sum of neighbouring pixeles (over thres) is over 0.5
    """
    data = (img * (img > 0)) / E
    photons = _np.floor(data)  # whole photons
    remainder = data - photons
    remainder_ismax = _np.logical_and(remainder == _snd.filters.maximum_filter(remainder, 3), remainder > thres)
    el = _snd.morphology.generate_binary_structure(2, 1)
    photons += _np.rint(_snd.filters.convolve(remainder, el) * remainder_ismax)  # sum over neighbours
    return photons


def photons_simple(img, E, ev_per_adu=3.65, bg=0):
    return _np.rint(((_np.squeeze(_np.array(img)) * ev_per_adu) - bg) / E)


def create_mask(img, lowthres=5, highthres=95, sigma=10, hotpixelstd=5):
    """
    create mask by high/low threshold (in percentile) in gaussian (with sigma) blurred image and morphologically cleaning.
    if hotpixelstd is not None, pixels higher then hotpixelstd times the standard deviation over masked mean will be ignored.
    """
    blured = _snd.gaussian_filter(img, sigma)
    mask = _np.logical_or.reduce((blured < _np.nanpercentile(blured, lowthres), blured > _np.nanpercentile(blured, highthres), _np.isnan(img),))
    sel21 = _snd.morphology.generate_binary_structure(2, 1)
    mask_cleaned = _snd.morphology.binary_dilation(mask, sel21, 2)
    mask_cleaned = _snd.morphology.binary_closing(mask_cleaned, sel21, 20)
    sel22 = _snd.morphology.generate_binary_structure(2, 2)
    mask_cleaned = _snd.morphology.binary_dilation(mask_cleaned, sel22, 2)
    mask = _np.logical_or(mask, mask_cleaned)

    if hotpixelstd is not None:
        hotpixel = img > (_np.mean(img[~mask]) + hotpixelstd * _np.std(img[~mask]))
        hotpixel = _snd.morphology.binary_dilation(hotpixel, sel22, 2)
        mask[hotpixel] = True
    return mask


def diffdist(*args):
    """
    returns Euclidean norm next neighbour difference of n coordinates: |diffdist(x,y,z)=diff(x),diff(y),diff(z)|
    """
    accum = 0
    for arg in args:
        accum += _np.diff(arg) ** 2
    return _np.sqrt(accum)


def atleastnd(array, n):
    """
    adds dimensions of length 1 in front of array to get to n dimensions
    """
    if not isinstance(array, _np.ndarray):
        array = _np.array(array)
    return array[tuple((n - array.ndim) * [None] + [...])]


def fastlen(x, factors=(2, 3, 5, 7, 11)):
    """
    return N>=x conisting only of the prime factors given as factors
    """
    # fmt: off
    fastlens = _np.array((
                            1,    2,    3,    4,    5,    6,    7,    8,    9,    10,   11,
                            12,   14,   15,   16,   18,   20,   21,   22,   24,   25,   27,
                            28,   30,   32,   33,   35,   36,   40,   42,   44,   45,   48,
                            49,   50,   54,   55,   56,   60,   63,   64,   66,   70,   72,
                            75,   77,   80,   81,   84,   88,   90,   96,   98,   99,   100,
                            105,  108,  110,  112,  120,  121,  125,  126,  128,  132,  135,
                            140,  144,  147,  150,  154,  160,  162,  165,  168,  175,  176,
                            180,  189,  192,  196,  198,  200,  210,  216,  220,  224,  225,
                            231,  240,  242,  243,  245,  250,  252,  256,  264,  270,  275,
                            280,  288,  294,  297,  300,  308,  315,  320,  324,  330,  336,
                            343,  350,  352,  360,  363,  375,  378,  384,  385,  392,  396,
                            400,  405,  420,  432,  440,  441,  448,  450,  462,  480,  484,
                            486,  490,  495,  500,  504,  512,  525,  528,  539,  540,  550,
                            560,  567,  576,  588,  594,  600,  605,  616,  625,  630,  640,
                            648,  660,  672,  675,  686,  693,  700,  704,  720,  726,  729,
                            735,  750,  756,  768,  770,  784,  792,  800,  810,  825,  840,
                            847,  864,  875,  880,  882,  891,  896,  900,  924,  945,  960,
                            968,  972,  980,  990,  1000, 1008, 1024, 1029, 1050, 1056, 1078,
                            1080, 1089, 1100, 1120, 1125, 1134, 1152, 1155, 1176, 1188, 1200,
                            1210, 1215, 1225, 1232, 1250, 1260, 1280, 1296, 1320, 1323, 1331,
                            1344, 1350, 1372, 1375, 1386, 1400, 1408, 1440, 1452, 1458, 1470,
                            1485, 1500, 1512, 1536, 1540, 1568, 1575, 1584, 1600, 1617, 1620,
                            1650, 1680, 1694, 1701, 1715, 1728, 1750, 1760, 1764, 1782, 1792,
                            1800, 1815, 1848, 1875, 1890, 1920, 1925, 1936, 1944, 1960, 1980,
                            2000, 2016, 2025, 2048, 2058, 2079, 2100, 2112, 2156, 2160, 2178,
                            2187, 2200, 2205, 2240, 2250, 2268, 2304, 2310, 2352, 2376, 2400,
                            2401, 2420, 2430, 2450, 2464, 2475, 2500, 2520, 2541, 2560, 2592,
                            2625, 2640, 2646, 2662, 2673, 2688, 2695, 2700, 2744, 2750, 2772,
                            2800, 2816, 2835, 2880, 2904, 2916, 2940, 2970, 3000, 3024, 3025,
                            3072, 3080, 3087, 3125, 3136, 3150, 3168, 3200, 3234, 3240, 3267,
                            3300, 3360, 3375, 3388, 3402, 3430, 3456, 3465, 3500, 3520, 3528,
                            3564, 3584, 3600, 3630, 3645, 3675, 3696, 3750, 3773, 3780, 3840,
                            3850, 3872, 3888, 3920, 3960, 3969, 3993, 4000, 4032, 4050, 4096,
                            4116, 4125, 4158, 4200, 4224, 4235, 4312, 4320, 4356, 4374, 4375
    )) # noqa
    # fmt: on
    if factors != (2, 3, 5, 7, 11) or _np.any(x > 4375):
        # slow fallback
        fastlens = _np.unique(
            [
                _np.product(_np.array(factors) ** _np.array(i).astype(float))
                for i in _it.product(*(range(int(2 + _np.log(_np.max(x)) / _np.log(k))) for k in factors))
            ]
        )
    return fastlens[_np.searchsorted(fastlens, x)].astype(int)


def split(x, dx, v=None):
    """
    splits an array x into parts dx apart
    x: 1d array
    dx: scalar
    v (optional): splits v along first axis as x would be split
    returns a list of parts of the array, with empty arrays for empty intervals
    """
    sid = _np.argsort(x)
    s = x[sid]
    ids = []
    last = 0
    for upper in _np.arange(0, s[-1], dx):
        last += _np.argmax(s[last:] > upper)
        ids.append(last)
    ids.append(len(s))
    if v is None:
        return [s[start:stop] for start, stop in zip(ids[0:-1], ids[1:])]
    else:
        return [v[sid[start:stop], ...] for start, stop in zip(ids[0:-1], ids[1:])]


def splits(N, sections, return_lists=False):
    """
    split N in sections ranges
    """
    each, remainder = divmod(N, sections)
    splits = _np.array(([0] + remainder * [each + 1] + (sections - remainder) * [each])).cumsum()
    ranges = [range(splits[i], splits[i + 1]) for i in range(sections)]
    if return_lists:
        return [list(i) for i in ranges]
    else:
        return ranges


def shortsci(number, decimals=0):
    """
    short scientific representation of number with variable precision and no plus or leading zero (!) in exponent as string
    """
    if number == 0:
        return '0e0'
    elif not _np.isfinite(number):
        return str(number)
    else:
        exponent = _np.floor(_np.log10(abs(number)))
        base = round(number / (10 ** exponent), decimals)
        if abs(base) >= 10:
            base = round(base / 10, decimals)
            exponent += 1
        return f'{base:.{decimals}f}e{int(exponent)}'


@_numba.njit
def axisrotation(axis, theta):
    """
    axis and angle to rotation matrix
    """
    u = axis / _np.linalg.norm(axis.astype(_np.float64))
    return _np.cos(theta) * _np.identity(3) + _np.sin(theta) * _np.cross(_np.identity(3), u) + (1 - _np.cos(theta)) * _np.outer(u, u)


@_numba.njit
def rotation(alpha, beta, gamma):
    """
    angles to rotation matrix
    """
    cosa, cosb, cosg = _np.cos(_np.array((alpha, beta, gamma)))
    sina, sinb, sing = _np.sin(_np.array((alpha, beta, gamma)))

    # yaw pitch roll
    M = _np.array(
        [
            [cosb * cosg, sina * sinb * cosg - cosa * sing, cosa * sinb * cosg + sina * sing],
            [cosb * sing, sina * sinb * sing + cosa * cosg, cosa * sinb * sing - sina * cosg],
            [-sinb, sina * cosb, cosa * cosb],
        ]
    )
    return M


def angles(rotmatrix):
    """
    rotation matrix to angles
    """
    return _np.arctan2(rotmatrix[2, 1], rotmatrix[2, 2]), -_np.arcsin(rotmatrix[2, 0]), _np.arctan2(rotmatrix[1, 0], rotmatrix[0, 0])


def gnorm(x, fwhm, rho, axis=None):
    """
    generalised normal distribution
    """
    s = _np.log(2) * (2 / fwhm) ** rho
    if axis is not None:
        return _np.exp(_ne.evaluate(f'sum(-abs(x)**rho*s), axis={axis})'))
    else:
        return _ne.evaluate('exp(-abs(x)**rho*s)')


def fwhm(X, Y):
    """
    finds the fwhm of y(x) without any interpolation
    """
    id1 = X[_np.argmax(Y > 0.5 * Y.max())]
    id2 = X[-_np.argmax(Y[::-1] > 0.5 * Y.max())]
    return id2 - id1


def arrayfromiter(iterable, count=-1):
    """
    converts an iterable of nd arrays to an n+1-d array
    """
    try:
        if count == -1:
            count = len(iterable)
    except TypeError:
        return _np.array(list(iterable))
    ret = None
    for i, el in enumerate(iterable):
        if ret is None:
            ret = _np.zeros((count, *_np.shape(el)))
        ret[i, ...] = _np.asarray(el)
    return ret


def indmax(array: _np.ndarray) -> tuple:
    """
    unravel_index(nanargmax(array))

    :param array: nd array
    :return: index of maximum
    """
    return _np.unravel_index(_np.nanargmax(array), array.shape)


def indmin(array: _np.ndarray) -> tuple:
    """
    unravel_index(nanargmin(array))

    :param array: nd array
    :return: index of maximum
    """
    return _np.unravel_index(_np.nanargmin(array), array.shape)


def intersect2d(a: _np.ndarray, b: _np.ndarray) -> _np.ndarray:
    """
    finds identical rows in array a,b

    :param a array1
    :param b array2
    :return array of intersecting rows
    """
    intersect = _np.intersect1d(a.view([('', a.dtype)] * a.shape[1]), b.view([('', b.dtype)] * b.shape[1]))
    return intersect.view(a.dtype).reshape(-1, a.shape[1])


def isdir(string):
    """
    Is string an existing dir?
    :param string: path
    :return: abspath or raise NotADirectoryError if is not a dir
    """
    import os

    if os.path.isdir(string):
        return os.path.abspath(string)
    else:
        raise NotADirectoryError(string)


def isfile(string):
    """
    Is string an existing file?
    :param string: path
    :return: abspath or raise FileNotFoundError if is not a dir
    """
    import os

    if os.path.isfile(string):
        return os.path.abspath(string)
    else:
        raise FileNotFoundError(string)


def alignedarray(shape, dtype=_np.float64, alignment=64, zero=False):
    """
    Create an aligned array
    :param shape: shape of the array
    :param dtype: dtype of the array
    :param alignment: alignment in bytes
    :param zero: if True, the array will be zero'd
    :returns: aligned array
    """
    dtype = _np.dtype(dtype)
    nbytes = _np.prod(shape) * dtype.itemsize
    buf = _np.empty(nbytes + alignment, dtype=_np.uint8)
    start_index = -buf.ctypes.data % alignment
    array = buf[start_index : start_index + nbytes].view(dtype).reshape(shape)
    if zero:
        array[:] = 0
    return array
