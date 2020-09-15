import numpy as _np
import numexpr as _ne
# from ..util import abs2



# def corr(input):
#     ret = _np.zeros([2 * s for s in input.shape])
#     ret[:input.shape[0], :input.shape[1]] = input
#     ret = _np.fft.fftshift(_np.fft.irfftn(abs2(_np.fft.rfftn(ret))))
#     return ret


def _fastlen(length):
        """
        gets fast (radix 2,3,5)fft length for at least length 
        """
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

        for fastlen in fastlens:
            if fastlen >= length:
                return fastlen
        return length

def corr(input):
    fftshape=[_fastlen(2 * s) for s in input.shape]
    f = (_np.fft.rfftn(input,s=fftshape))
    _ne.evaluate('(f*conj(f))',out=f)
    ret = _np.fft.irfftn(f)
    ret = _np.fft.fftshift(ret)[tuple((slice(ps//2 - s , ps//2 + s) for s, ps in zip(input.shape, fftshape)))]
    return ret
