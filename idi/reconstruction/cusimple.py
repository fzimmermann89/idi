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
        tuple(
            (
                Ellipsis,
                *(slice(ps // 2 - input.shape[ax], ps // 2 + input.shape[ax]) for ax, ps in zip(axes, fftshape)),
            )
        )
    ]
    if norm:
        n = corr(_cp.ones(tuple(input.shape[ax] for ax in axes)), returngpu=True)
        ret /= n
        ret[(...,) + (n < 0.9).nonzero()] = _np.nan
    if not returngpu:
        ret = _cp.asnumpy(ret)
        _cp.get_default_memory_pool().free_all_blocks()
    return ret


_div = _cp.ElementwiseKernel("T x, T count", "T res", "if(count>0.5){res = x/count;} else {res = 0;}", "div")


class correlator2d:
    def __init__(self, mask=None, norm=True) -> None:
        """Simple 2D correlator
        Does the 2D autocorrelation divided by the mask autocorrelation.
        mask: mask to use for the autocorrelation or None for all-ones mask
        norm: normalize input by mean over all pixels.
        """

        self._fftshape = None
        if mask is not None:
            mask = _cp.asarray(mask, dtype=_cp.float32)
            self._maskcorr = self._corr(mask)
            self._mask = mask
        else:
            self._mask = None
            self._maskcorr = None
        self._mean = None
        self._n = 0
        self._nvar = None
        self._norm = norm

    def _corr(self, input: _cp.array) -> _cp.array:
        if self._fftshape is None:
            self._fftshape = [_fastlen(2 * s) for s in input.shape]
        ret = _cp.fft.rfft2(input, s=self._fftshape)
        ret = _cp.abs(ret) ** 2
        ret = _cp.fft.irfft2(ret)
        ret = _cp.fft.fftshift(ret, axes=(-2, -1))[
            ...,
            slice(self._fftshape[-2] // 2 - input.shape[-2], self._fftshape[-2] // 2 + input.shape[-2]),
            slice(self._fftshape[-1] // 2 - input.shape[-1], self._fftshape[-1] // 2 + input.shape[-1]),
        ]
        return ret

    def add(self, input: _np.array, mask=None) -> None:
        if input.ndim != 2:
            raise ValueError("input must be 2D.")
        input = _cp.array(input, dtype=_cp.float32, copy=True)
        if self._norm:
            input /= _cp.mean(input)

        if mask is not None:
            input *= _cp.asarray(mask, dtype=_cp.float32)
            maskcorr = self._corr(mask)
        elif self._mask is not None:
            input *= self._mask
            maskcorr = self._maskcorr
        elif self._maskcorr is not None:
            maskcorr = self._maskcorr
        else:
            mask = _cp.ones_like(input)
            maskcorr = self._corr(mask)
            self._maskcorr = maskcorr

        corr = self._corr(input)
        corr = _div(corr, maskcorr)

        if self._mean is None:
            self._mean = corr
            self._n = 1
            self._nvar = _cp.zeros_like(corr)
        else:
            self._n += 1
            delta = corr - self._mean
            self._mean += delta / self._n
            delta2 = corr - self._mean
            self._nvar += delta * delta2

    @property
    def mean(self) -> _np.array:
        return self._mean.get()

    @property
    def n(self) -> int:
        return self._n

    @property
    def std(self) -> _np.array:
        return _cp.sqrt(self._nvar / self._n).get()
