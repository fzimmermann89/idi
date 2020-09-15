import numpy as _np
from ..util import abs2

def corr(input):
    ret = _np.zeros([2 * s for s in input.shape])
    ret[:input.shape[0], :input.shape[1]] = input
    ret = _np.fft.fftshift(_np.fft.irfftn(abs2(_np.fft.rfftn(ret))))
    return ret