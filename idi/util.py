from __future__ import division as _future_division, print_function as _future_print
import numba as _numba
import numpy as _np
import scipy.ndimage as _snd
import numexpr as _ne

def create_mask(img, lowthres=50, highthres=95, sigma=10):
    '''
    create mask by high/low threshold in blurred image. WIP
    '''
    blured=_snd.gaussian_filter(img,sigma)
    blured[img==0]=_np.nan
    #blured[img<_np.nanpercentile(blured,5)]=_np.nan
    low=blured<=_np.nanpercentile(blured,lowthres)
    high=blured>=_np.nanpercentile(blured,highthres)
    mask=_np.logical_or(high,low)
    mask[img==0]=True
    mask_cleaned=_snd.morphology.binary_dilation(mask,_snd.morphology.generate_binary_structure(2,1),2)
    mask_cleaned=_snd.morphology.binary_closing(mask_cleaned,_snd.morphology.generate_binary_structure(2,1),20)
    mask_cleaned=_snd.morphology.binary_dilation(mask_cleaned,_snd.morphology.generate_binary_structure(2,2),2)
    mask=_np.logical_or(mask,mask_cleaned)
    
    #hotpixel
    hotpixel=img>(_np.mean(img[~mask])+5*_np.std(img[~mask]))
    hotpixel=_snd.morphology.binary_dilation(hotpixel,_snd.morphology.generate_binary_structure(2,2),2)
    mask[hotpixel]=True
    return mask

def diffdist(*args):
    '''
    returns Euclidean norm next neighbour difference of n coordinates: |diffdist(x,y,z)=diff(x),diff(y),diff(z)|
    '''
    accum = 0
    for arg in args:
        accum += _np.diff(arg) ** 2
    return _np.sqrt(accum)


def radial_profile(data, center, calcStd=False, os=1):
    '''
    calculates a ND radial profile of data around center. will ignore nans
    calStd: calculate standard deviation, return tuple of (profile, std)
    os: oversample by a factor. With default 1 the stepsize will be 1 pixel, with 2 it will be .5 pixels etc. 
    '''

    if len(center) != data.ndim:
        raise TypeError('center should be of length data.ndim')
    center = _np.array(center)[tuple([slice(len(center))] + data.ndim * [None])]
    ind = _np.indices((data.shape))
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
    ind0 = ~_np.all(_np.isnan(array), axis=0)
    ind1 = ~_np.all(_np.isnan(array), axis=1)
    return array[ind1, :][:, ind0]


def rebin(arr, n):
    #deprecated
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
    '''
    Return the center newshape portion of the array.
    '''
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
    '''
    fill invalid values by closest valid value. invalid: mask of invalid values, default: np.isnan(data)
    '''
    if invalid is None:
        invalid = _np.isnan(data)
    ind = _snd.distance_transform_edt(invalid, return_distances=False, return_indices=True)
    return data[tuple(ind)]


def photons_localmax(img, E, thres=0.):
    '''
    photonize image. First count whole photons. Second count fractional/split photons at local maxima if sum of neighbouring pixeles (over thres) is over 0.5
    '''
    data = (img * (img>0)) / E
    photons = _np.floor(data) #whole photons
    remainder = data - photons
    remainder_ismax = _np.logical_and(remainder == _snd.filters.maximum_filter(remainder, 3), remainder>thres)
    el = _snd.morphology.generate_binary_structure(2, 1)
    photons += _np.rint(_snd.filters.convolve(rest, el) * rest_ismax) #sum over neighbours
    return photons

def photons_simple(img, E, ev_per_adu=3.65, bg=0):
    return _np.rint(((_np.squeeze(_np.array(img)) * ev_per_adu) - bg) / E)


class accumulator:
    '''
    simple accumulator for large number of data points. allows retrieval of mean and stdev
    '''
    def __init__(self, like=None):
        self._n = 0
        if like is None:
            self._mean = None
            self._nvar = None
        else:
            self._mean = _np.zeros_like(like)
            self._nvar = _np.zeros_like(like)

    def __repr__(self):
        print(type(self._mean))
        return 'accumulator[%i]' % self._n
    
    def add(self, value, count=1):
        self._n += count
        if self._mean is None:
            self._mean = _np.asarray(value).astype(_np.float64)
            self._nvar = _np.zeros_like(value).astype(_np.float64)
        else:
            with _np.errstate(divide='ignore', invalid='ignore'):
                delta = value - self._mean
                self._mean = _np.add(self._mean, delta / self._n, where=(count!=0), out=self._mean)
                self._nvar = _ne.evaluate(
                    'nvar + delta * (value - mean)',
                    local_dict={'nvar': self._nvar, 'value': value, 'delta': delta, 'mean': self._mean},
                )

    def __len__(self):
        return _np.max(self._n)
    
    @property
    def shape(self):
        if self._mean is not None:
            return _np.asarray(self._mean).shape
        else:
            return 0
    @property
    def mean(self):
        if self._mean is not None:
            return self._mean
        else:
            return 0

    @property
    def var(self):
        if self._nvar is not None:
            return self._nvar / self._n
        else: 
            return 0

    @property
    def std(self):
        return _np.sqrt(self.var)

    



def filter_std(image, size, sigma=1):
    '''
    explicitly filters an nd image based on deviation by local stdev over local mean.
    return mask of points that are sigma local stdev over(under) the local mean if sigma is positiv(negativ)
    '''
    import numpy as np
    import ctypes
    import scipy.ndimage as ndi
    from numba import cfunc, carray
    from numba.types import intc, intp, float64, voidptr, CPointer
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
    res = image>ndi.generic_filter(image, LowLevelCallable(_std.ctypes, ptr), size)
    if sigma<0: res = ~res
    return res



import scipy.signal as _ss
def fftfilter_mean(image,size, norm=False):
    strel=_np.ones(image.ndim*[size])/size**image.ndim
    res=_ss.fftconvolve(image,strel,mode='same')
    if norm: res/=_ss.fftconvolve(_np.ones_like(image),strel,mode='same')
    return res

def fftfilter_std(image,size,sigma):
    norm=fftfilter_mean(_np.ones_like(image),size)
    mean=fftfilter_mean(image,size)/norm
    sqmean=fftfilter_mean(image*image,size)/norm
    std=_np.nan_to_num(_np.sqrt(sqmean-mean*mean))
    res=_np.abs(image-mean)>sigma*std
    res[_np.abs(sigma*std)<1e-15]=0
    return res