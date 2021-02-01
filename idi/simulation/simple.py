import numpy as _np
from ..util import rebin as _rebin


def simulate(simobject, Ndet, pixelsize, detz, k, fftfunction=_np.fft.fft2):
    '''
    Simple (fft based) Fraunhofer small angle approximation
    '''
    if _np.size(Ndet) == 1:
        Ndet = [Ndet, Ndet]
    Ndet = _np.array(Ndet)
    dq = k * pixelsize / detz
    maxq = Ndet / 2 * dq
    dx = _np.pi / maxq
    img = simobject.getImg(dx)
    os = _np.ceil(np.array(img.shape) / Ndet).astype(int)
    return _rebin(_np.fft.fftshift(fftfunction(a, s=os * Ndet)), os, 'mean')