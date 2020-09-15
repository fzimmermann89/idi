import numpy as _np
import numexpr as _ne
 
class correlator:
    def __init__(self, mask, fftfunctions=(_np.fft.rfftn, _np.fft.irfftn)):
        """
        nd-correlations with constant mask. 
        
        will not normalize between different added images, but each single image. 
        optional fftfunctions argument is pair of forward and backward fft function to use (default: numpy default)
        based on http://www.dirkpadfield.com/Home/MaskedFFTRegistrationPresentation.pdf but instead of subtracting correlations of mean, divide by it
        masked Autocorrelation of Image =          (maskedImage C maskedImage)
                                         --------------------------------------------
                                                (mask C Image) * (Image C mask)
                                                         -----------
                                                         (mask C mask)
        """
        fft, ifft = fftfunctions
        self._fft = fftfunctions
        self._shape = mask.shape
        self._padshape = tuple((correlator._fastlen(2 * s) for s in mask.shape))
        self._mask = self._pad(mask)
        self._fmask = fft(self._mask)
        self._mCm = ifft(self._fmask * self._fmask.conj())
        
    def corr(self, image):
        """
        does a new correlation
        """
        for ds, cs in zip(image.shape, self._shape):
            if ds != cs:
                raise ValueError('data has not expected shape')
        fft, ifft = self._fft
        pimage = self._pad(image * self._mask[tuple((slice(0, s) for s in self._shape))])
        fimg = fft(pimage)
        res = ifft(_ne.evaluate('fimg*conj(fimg)')) #iCi
        _ne.evaluate('fimg*conj(fmask)',local_dict={'fmask':self._fmask,'fimg':fimg},out=fimg)
        norm = ifft(fimg) #iCm
        _ne.evaluate('conj(fimg)',out=fimg)
        # after fftshift it would be mCi=flip iCm in both directions, 
        #fftshift effects first row/column differently, so in the unshifted version a multiplication by mCi is:
        #norm *= _np.roll(norm[tuple(norm.ndim * [slice(None, None, -1)])], norm.ndim * [1], range(0, norm.ndim))
        norm *= ifft(fimg)
        _ne.evaluate('where(norm>1e-5,res/norm*mCm,res*mCm)',out=res,local_dict={'res':res,'norm':norm,'mCm':self._mCm})
        res=_np.fft.fftshift(res)[tuple((slice(ps//2 - s + 1, ps//2 + s) for s, ps in zip(self._shape, self._padshape)))]
        return res

    
    
    
    def _pad(self, data):
        """
        pads data to size, data will be in top left corner of return
        """
        ret = _np.zeros(self._padshape, _np.float64)
        ret[tuple((slice(0, s) for s in self._shape))] = data
        return ret

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

    @property
    def shape_input(self):
        """
        expected input shape for add function
        """
        return self._shape

    @property
    def shape_result(self):
        """
        shape of the result
        """
        return tuple((2 * s - 1 for s in self._shape))

    @property
    def mask(self):
        """
        used mask
        """
        return self._mask[tuple((slice(0, s) for s in self._shape))]
    
 

 