import numpy as _np
import scipy.signal as _ss


def filter_std(image, size, sigma=1):
    """
    explicitly filters an nd image based on deviation by local stdev over local mean.
    return mask of points that are sigma local stdev over(under) the local mean if sigma is positiv(negativ)
    """
    import numpy as np
    import ctypes
    import scipy.ndimage as ndi
    from numba import cfunc, carray
    from numba.core.types import intc, intp, float64, voidptr, CPointer
    from scipy import LowLevelCallable

    @cfunc(intc(CPointer(float64), intp, CPointer(float64), voidptr), fastmath=True)
    def _std(values_ptr, len_values, result, data):
        values = carray(values_ptr, (len_values,), dtype=float64)
        accumx = 0
        accumx2 = 0
        sigma = carray(data, (1,), dtype=float64)[0]
        for x in values:
            accumx += x
            accumx2 += x * x
        mean = accumx / len_values
        std = np.sqrt((accumx2 / len_values) - mean ** 2)
        result[0] = sigma * std + mean
        return 1

    csigma = ctypes.c_double(sigma)
    ptr = ctypes.cast(ctypes.pointer(csigma), ctypes.c_void_p)
    res = image > ndi.generic_filter(image, LowLevelCallable(_std.ctypes, ptr), size)
    if sigma < 0:
        res = ~res
    return res


def fftfilter_mean(image, size, norm=False):
    strel = _np.ones(image.ndim * [size]) / size ** image.ndim
    res = _ss.fftconvolve(image, strel, mode='same')
    if norm:
        res /= _ss.fftconvolve(_np.ones_like(image), strel, mode='same')
    return res


def fftfilter_std(image, size, sigma):
    norm = fftfilter_mean(_np.ones_like(image), size)
    mean = fftfilter_mean(image, size) / norm
    sqmean = fftfilter_mean(image * image, size) / norm
    std = _np.nan_to_num(_np.sqrt(sqmean - mean * mean))
    res = _np.abs(image - mean) > sigma * std
    res[_np.abs(sigma * std) < 1e-15] = 0
    return res
