import numpy as _np
from . import autocorrelate3


def fastlen(length):
    fastlens = (
        8, 9, 10, 12, 15, 16, 18, 20, 24, 25, 27, 30, 32, 36, 40, 45, 48, 50, 54, 60, 64, 72, 75, 80, 81, 90, 96, 100, 108, 120, 
        125, 128, 135, 144, 150, 160, 162, 180, 192, 200, 216, 225, 240, 243, 250, 256, 270, 288, 300, 320, 324, 360, 375, 384, 
        400, 405, 432, 450, 480, 486, 500, 512, 540, 576, 600, 625, 640, 648, 675, 720, 729, 750, 768, 800, 810, 864, 900, 960, 
        972, 1000, 1024, 1080, 1125, 1152, 1200, 1215, 1250, 1280, 1296, 1350, 1440, 1458, 1500, 1536, 1600, 1620, 1728, 1800, 
        875, 1920, 1944, 2000, 2025, 2048, 2160, 2187, 2250, 2304, 2400, 2430, 2500, 2560, 2592, 2700, 2880, 2916, 3000, 3072, 
        3125, 3200, 3240, 3375, 3456, 3600, 3645, 3750, 3840, 3888, 4000, 4050, 4096, 4320, 4374, 4500, 4608, 4800, 4860, 5000, 
        5120, 5184, 5400, 5625, 5760, 5832, 6000, 6075, 6144, 6250, 6400, 6480, 6561, 6750, 6912, 7200, 7290, 7500, 7680, 7776, 
        8000, 8100, 8192, 8640, 8748, 9000, 9216, 9375, 9600, 9720, 10000
    )
    
    for l in fastlens:
        if l > length: return l
    return length


def prepare(input, z):
    y, x = _np.meshgrid(_np.arange(input.shape[0], dtype=_np.float64), _np.arange(input.shape[1], dtype=_np.float64))
    x -= input.shape[0] / 2.0
    y -= input.shape[1] / 2.0
    d = _np.sqrt(x ** 2 + y ** 2 + z ** 2)
    qx, qy, qz = [(k / d * z) for k in (x, y, z)]
    #     kx1=kx1/(kx1[1,0]-kx1[0,0]) #correct, but slower
    #     ky1=ky1/(ky1[0,1]-ky1[0,0])
    qx, qy, qz = [_np.rint(k - _np.min(k)).astype(int, copy=False) for k in (qx, qy, qz)]
    qlenx, qleny, qlenz = [fastlen(2 * (_np.max(k) + 1)) for k in (qx, qy, qz)]
    ret = _np.zeros((qlenz, qleny, qlenx), dtype=_np.float64)
    _np.add.at(ret, (qz, qy, qx), input)
    #     ret[kz1,ky1,kx1]=input #only if no double assignment
    return ret


def _corr(input, z):
    tmp = prepare(input, z)
    autocorrelate3.autocorrelate3(tmp)
    return tmp[: tmp.shape[0] // 2, ...]


def corr(input, z):
    if input.ndim == 2:
        return _corr(input, z)
    elif input.ndim == 3:
        s = prepare(_np.zeros_like(input[0, ...]), z).shape
        res = _np.zeros((s[0] // 2, s[1], s[2]))
        for inp in input:
            _np.add(res, _corr(inp, z), out=res)
        return res
    else:
        raise TypeError
