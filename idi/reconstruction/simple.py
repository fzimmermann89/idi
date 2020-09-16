import numpy as _np
import numexpr as _ne
# from ..util import abs2
from ..util import fastlen as _fastlen


# def corr(input):
#     ret = _np.zeros([2 * s for s in input.shape])
#     ret[:input.shape[0], :input.shape[1]] = input
#     ret = _np.fft.fftshift(_np.fft.irfftn(abs2(_np.fft.rfftn(ret))))
#     return ret



def corr(input):
    fftshape=[_fastlen(2 * s) for s in input.shape]
    f = (_np.fft.rfftn(input,s=fftshape))
    _ne.evaluate('(f*conj(f))',out=f)
    ret = _np.fft.irfftn(f)
    ret = _np.fft.fftshift(ret)[tuple((slice(ps//2 - s , ps//2 + s) for s, ps in zip(input.shape, fftshape)))]
    return ret
