import numpy as _np
import cupy as _cp
from ..util import fastlen as _fastlen


def corr(input, axes=(-1, -2), norm=False, returngpu=False, **kwargs):
    """
    simple autocorrelation of input along axes (default: last two) using gpu
    axes: axes to correlate along, defaults to last two
    norm: do normalisation along non correlation axes and normalise for pair count
    returngpu: retrun a cupy array
    """

    axes = sorted([input.ndim + a if a < 0 else a for a in axes])
    fftshape = [_fastlen(2 * input.shape[ax]) for ax in axes]
    dinput = _cp.array(input)
    if norm:
        dinput *= 1 / dinput.mean(axis=[i for i in range(input.ndim) if i not in axes] or None)
    ret = _cp.fft.rfftn(dinput, fftshape)
    ret = _cp.abs(ret) ** 2
    ret = _cp.fft.irfftn(ret, axes=axes)
    ret = _cp.fft.fftshift(ret, axes=axes)[
        tuple((Ellipsis, *(slice(ps // 2 - input.shape[ax], ps // 2 + input.shape[ax]) for ax, ps in zip(axes, fftshape))))
    ]
    if norm:
        n = corr(_cp.ones(tuple(input.shape[ax] for ax in axes)), returngpu=True)
        ret /= n
        ret[(...,) + (n < 0.9).nonzero()] = _np.nan
    if not returngpu:
        ret = _cp.asnumpy(ret)
        _cp.get_default_memory_pool().free_all_blocks()
    return ret
