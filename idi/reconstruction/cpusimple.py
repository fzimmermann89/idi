import numpy as _np
from ..util import fastlen as _fastlen


def corr(input, axes=(-2, -1), norm=False, fftfunctions=(_np.fft.rfftn, _np.fft.irfftn), **kwargs):
    """
    simple autocorrelation of input along axes (default: last two)
    axes: axes to correlate along, defaults to last two
    norm: do normalisation along non correlation axes and normalise for pair count
    """
    fft, ifft = fftfunctions
    fftshape = [_fastlen(2 * input.shape[ax]) for ax in axes]
    fftaxes = None if set(axes) == {-1, -2} and input.ndim == 2 else axes
    if norm:
        input = input * (1 / input.mean(axis=tuple([i for i in range(input.ndim) if i not in axes]) or None))
    ret = fft(input, fftshape, axes=fftaxes)
    ret = ret * ret.conj()
    # _ne.evaluate('(ret*conj(ret))', out=ret, casting='same_kind')
    ret = ifft(ret, axes=fftaxes)
    ret = _np.fft.fftshift(ret, axes=axes)[
        tuple((Ellipsis, *(slice(ps // 2 - input.shape[ax], ps // 2 + input.shape[ax]) for ax, ps in zip(axes, fftshape))))
    ]

    if norm:
        n = corr(_np.ones(tuple(input.shape[ax] for ax in axes)), fftfunctions=fftfunctions)
        ret /= n
        ret[(...,) + (n < 0.9).nonzero()] = _np.nan
    return ret


class correlator2d:
    def __init__(self, mask=None, norm: bool = True) -> None:
        """Simple 2D correlator
        Does the 2D autocorrelation divided by the mask autocorrelation.
        mask: mask to use for the autocorrelation or None for all-ones mask
        norm: normalize input by mean over all pixels.
        """
        self._fftshape = None
        if mask is not None:
            mask = _np.asarray(mask, dtype=_np.float32)
            self._maskcorr = self._corr(mask)
            self._mask = mask
        else:
            self._mask = None
            self._maskcorr = None
        self._mean = None
        self._n = 0
        self._nvar = None
        self._norm = norm

    def _corr(self, input: _np.ndarray) -> _np.ndarray:
        if self._fftshape is None:
            self._fftshape = [_fastlen(2 * s) for s in input.shape]
        ret = _np.fft.rfft2(input, s=self._fftshape)
        ret = _np.abs(ret) ** 2
        ret = _np.fft.irfft2(ret)
        ret = _np.fft.fftshift(ret, axes=(-2, -1))[
            ...,
            slice(self._fftshape[-2] // 2 - input.shape[-2], self._fftshape[-2] // 2 + input.shape[-2]),
            slice(self._fftshape[-1] // 2 - input.shape[-1], self._fftshape[-1] // 2 + input.shape[-1]),
        ]
        return ret

    def add(self, input: _np.ndarray, mask=None) -> None:
        if input.ndim != 2:
            raise ValueError("input must be 2D.")
        input = _np.array(input, dtype=_np.float32, copy=True)
        if self._norm:
            input /= _np.mean(input)
        if mask is not None:
            input *= mask
            maskcorr = self._corr(mask)
        elif self._mask is not None:
            input *= self._mask
            maskcorr = self._maskcorr
        elif self._maskcorr is not None:
            maskcorr = self._maskcorr
        else:
            mask = _np.ones_like(input)
            maskcorr = self._corr(mask)
            self._maskcorr = maskcorr

        corr = _np.divide(self._corr(input), maskcorr, where=maskcorr > 0.5, out=_np.zeros_like(maskcorr))

        if self._mean is None:
            self._mean = corr
            self._n = 1
            self._nvar = _np.zeros_like(corr)
        else:
            self._n += 1
            delta = corr - self._mean
            self._mean += delta / self._n
            delta2 = corr - self._mean
            self._nvar += delta * delta2

    @property
    def mean(self) -> _np.ndarray:
        return self._mean

    @property
    def n(self) -> int:
        return self._n

    @property
    def std(self) -> _np.ndarray:
        return _np.sqrt(self._nvar / self._n)
