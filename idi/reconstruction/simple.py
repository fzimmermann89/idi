import numpy as np


def corr(i_nput):
    ret = _np.zeros([2 * s for s in i_nput.shape])
    ret[: i_nput.shape[0], : i_nput.shape[1]] = i_nput
    ret = _np.fft.fftshift(_np.abs(_np.fft.irfftn(_np.square(_np.abs(_np.fft.rfftn(ret))))))
    return ret
