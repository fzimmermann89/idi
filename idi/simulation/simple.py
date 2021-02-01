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
    obj = simobject.getImg(dx)
    os = _np.ceil(np.array(img.shape) / Ndet).astype(int)
    img = _np.fft.fftshift(fftfunction(obj, s=os * Ndet))
    return _rebin(img, os, 'mean')


def simulate3d(simobject, Ndet, pixelsize, detz, k, fftfunction=_np.fft.fft2):
    '''
    Simple (fft based, 3d sliced) Fraunhofer small angle approximation
    '''
    if _np.size(Ndet) == 1:
        Ndet = [Ndet, Ndet]
    Ndet = _np.array(Ndet)
    dq = k * pixelsize / detz
    maxq = Ndet / 2 * dq
    dx = _np.pi / maxq
    dz = min(dx)
    obj = simobj.getImg(dx.tolist() + [dz], 3)
    os = _np.ceil(np.array(obj.shape[:2]) / Ndet).astype(int)
    qz = 1 - _np.sqrt(
        1 - sum(q ** 2 for q in _np.meshgrid(*[(pixelsize / (_detz * o)) * _np.arange(-n * o / 2, n * o / 2) for n, o in zip(Ndet, os)]))
    )
    f = np.fft.fftshift(fftfunction(obj, s=os * Ndet, axes=(0, 1)), axes=(0, 1))
    img = sum(f[..., i] * _np.exp((-1j * dz * k * i) * qz) for i in range(f.shape[-1]))
    return _rebin(img, os, 'mean')
