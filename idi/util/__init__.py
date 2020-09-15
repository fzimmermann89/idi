from __future__ import division as _future_division, print_function as _future_print

from .accum import *
from .filters import *
from .functools import *
from .poissondisk import *

import numba as _numba
import numpy as _np
import scipy.ndimage as _snd
import numexpr as _ne


def radial_profile(data, center=None, calcStd=False, os=1):
    '''
    calculates a ND radial profile of data around center. will ignore nans
    calStd: calculate standard deviation, return tuple of (profile, std)
    os: oversample by a factor. With default 1 the stepsize will be 1 pixel, with 2 it will be .5 pixels etc. 
    '''
    if center is None:
        center=_np.array(data.shape)//2
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
    '''
    remove full-nan rows and columns of 2d array
    '''
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
    photons += _np.rint(_snd.filters.convolve(remainder, el) * remainder_ismax) #sum over neighbours
    return photons



def photons_simple(img, E, ev_per_adu=3.65, bg=0):
    return _np.rint(((_np.squeeze(_np.array(img)) * ev_per_adu) - bg) / E)


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
