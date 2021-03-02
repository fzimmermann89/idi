import numpy as _np
import numexpr as _ne
from ..util import fastlen as _fastlen


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
        self._padshape = tuple(_fastlen(2 * s) for s in mask.shape)
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
        res = ifft(_ne.evaluate('fimg*conj(fimg)'))  # iCi
        _ne.evaluate('fimg*conj(fmask)', local_dict={'fmask': self._fmask, 'fimg': fimg}, out=fimg)
        norm = ifft(fimg)  # iCm
        _ne.evaluate('conj(fimg)', out=fimg)
        # after fftshift it would be mCi=flip iCm in both directions,
        # fftshift effects first row/column differently, so in the unshifted version a multiplication by mCi is:
        # norm *= _np.roll(norm[tuple(norm.ndim * [slice(None, None, -1)])], norm.ndim * [1], range(0, norm.ndim))
        norm *= ifft(fimg)
        _ne.evaluate('where(norm>1e-5,res/norm*mCm,res*mCm)', out=res, local_dict={'res': res, 'norm': norm, 'mCm': self._mCm})
        res = _np.fft.fftshift(res)[tuple((slice(ps // 2 - s + 1, ps // 2 + s) for s, ps in zip(self._shape, self._padshape)))]
        return res

    def _pad(self, data):
        """
        pads data to size, data will be in top left corner of return
        """
        ret = _np.zeros(self._padshape, _np.float64)
        ret[tuple((slice(0, s) for s in self._shape))] = data
        return ret

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
