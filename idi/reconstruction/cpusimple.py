import numpy as _np
from ..util import fastlen as _fastlen

# import numexpr as _ne
# from ..util import abs2
# def corr(input):
#     ret = _np.zeros([2 * s for s in input.shape])
#     ret[:input.shape[0], :input.shape[1]] = input
#     ret = _np.fft.fftshift(_np.fft.irfftn(abs2(_np.fft.rfftn(ret))))
#     return ret


def corr(input, axes=(-1, -2), norm=False, fftfunctions=(_np.fft.rfftn, _np.fft.irfftn), **kwargs):
    """
    simple autocorrelation of input along axes (default: last two)
    axes: axes to correlate along, defaults to last two
    norm: do normalisation along non correlation axes and normalise for pair count
    """
    fft, ifft = fftfunctions
    axes = sorted([input.ndim + a if a < 0 else a for a in axes])
    fftshape = [_fastlen(2 * input.shape[ax]) for ax in axes]
    if norm:
        input = input * (1 / input.mean(axis=[i for i in range(input.ndim) if i not in axes] or None))
    ret = fft(input, fftshape)
    ret = _np.abs(ret) ** 2
    # _ne.evaluate('(ret*conj(ret))', out=ret, casting='same_kind')
    ret = ifft(ret, axes=axes)
    ret = _np.fft.fftshift(ret, axes=axes)[
        tuple((Ellipsis, *(slice(ps // 2 - input.shape[ax], ps // 2 + input.shape[ax]) for ax, ps in zip(axes, fftshape))))
    ]

    if norm:
        n = corr(_np.ones(tuple(input.shape[ax] for ax in axes)), fftfunctions=fftfunctions)
        ret /= n
        ret[(...,) + (n < 0.9).nonzero()] = _np.nan
    return ret
