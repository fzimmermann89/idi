import numpy as _np
import numba as _numba
import numexpr as _ne
from ..util import fastlen, atleastnd, intersect2d, alignedarray
import itertools as _it
import warnings as _w

try:
    from .autocorrelate3 import autocorrelate3
except ImportError:
    _w.warn('using numpy ft')

    def autocorrelate3(data):
        tmp = _np.fft.rfftn(data[..., :-2])
        tmp *= tmp.conj()
        data[..., :-2] = _np.fft.irfftn(tmp, s=_np.add(data.shape, (0, 0, -2)))
        data[..., -2:] = 0
        return 0


def corr(input, z, sampling=None, verbose=False):
    """
    calculated 3d correlation of 2d input array sampled at distance z using fft.
    assumes center of input to be center of image
    if input is 3d, the result will be the sum along the first dimension.
    Parameters:
    z: Detector distance in pixels
    sampling: q resoltion - None (default) -> use center pixel q size
                          - 0: smallest step in qx and qy
                          - number -> centerpixel / number
                          - 3 numbers -> seperate centerpixel/number in qx,qy,qz
    """

    def _prepare(input, z, sampling):
        """
        transform centered 2d input sampled at distance z to 3d k-space
        """

        y, x = _np.meshgrid(_np.arange(input.shape[1], dtype=_np.float64), _np.arange(input.shape[0], dtype=_np.float64))
        x -= input.shape[0] / 2.0
        y -= input.shape[1] / 2.0
        d = _np.sqrt(x ** 2 + y ** 2 + z ** 2)
        qx, qy, qz = [(k / d * z) for k in (x, y, z)]

        if sampling is not None:
            sampling = _np.array(sampling)
            if sampling.size == 1:
                if sampling == 0:
                    # (more) correct, but slower. should think about correct oversampling
                    sampling = 1 / min(abs(qy[1, 1] - qy[0, 0]), abs(qx[1, 1] - qx[0, 0]))
                qx, qy, qz = ((k * sampling) for k in (qx, qy, qz))
            elif sampling.size == 3:
                qx, qy, qz = ((k * s) for k, s in zip((qx, qy, qz), sampling))
            else:
                raise ValueError('sampling should be None or 1 or 3 numbers')

        qx, qy, qz = [_np.rint(k - _np.min(k)).astype(int, copy=False) for k in (qx, qy, qz)]
        qlenx, qleny, qlenz = [fastlen(2 * (_np.max(k) + 1)) for k in (qx, qy, qz)]
        ret = alignedarray((qlenz, qleny, qlenx + 2), dtype=_np.float64, alignment=64, zero=True)  # additonal padding in qx for inplace fft
        _np.add.at(ret, (qz, qy, qx), input)
        #     ret[kz1,ky1,kx1]=input #only if no double assignment
        return ret

    def _corr(input, z, sampling):
        """
        wrapper for autocorrelate3, removes redundant slices in first dimension
        """
        tmp = _prepare(input, z, sampling)
        err = autocorrelate3(tmp)
        if err:
            raise RuntimeError(f'cython autocorrelations failed with error code {err}')
        return tmp[: tmp.shape[0] // 2, ...]

    if input.ndim == 2:
        if verbose:
            print('.', end=' ', flush=True)
        return _corr(input, z, sampling)
    elif input.ndim == 3:
        s = _prepare(_np.zeros_like(input[0, ...]), z, sampling).shape
        res = _np.zeros((s[0] // 2, s[1], s[2]))
        for n, inp in enumerate(input):
            if verbose:
                print(n, end=' ', flush=True)
            _np.add(res, _corr(inp, z), out=res)
        return res
    else:
        raise TypeError


def unwrap(img):
    """
    unwraps a single correlation result
    """
    return _np.roll(img[..., :-2], shift=(img.shape[1] // 2, (img.shape[2] - 2) // 2), axis=(1, 2))


class correlator_tiles:
    def __init__(self, qs, maskout, meandata):
        """
        3d fft correlator for tiled data
        qs: q vectors of data points, array of shape (T,P,3) with T: non overlapping tiles, P: pixels per tile
        maskout: points on detector that will not be used, array of shape (T,P)
        meandata: mean of data values used for normalisation, array of shape (T,P)
        resolution is fixed at dq=1
        """
        q = atleastnd(qs, 3)
        for a, b in _it.combinations(range(q.shape[0]), 2):
            if len(intersect2d(q[a, :, :], q[b, :, :])):
                _w.warn('qs are assumed to be unique in first dimension!')
        if not q.shape[2] == 3:
            raise ValueError('q should qz,qy,qx in last axis')
        mask = atleastnd(maskout, 2)
        mean = atleastnd(meandata, 2)
        if not q.shape[:2] == mask.shape == mean.shape:
            print(q.shape, mask.shape, mean.shape)
            raise ValueError('shape missmatch')

        q = _np.rint(q - _np.min(q[~maskout], axis=0)).astype(int)
        q[mask] = _np.max(q, axis=(0, 1)) + 2
        qlen = _np.array([fastlen(2 * (_np.max(k) + 1)) for k in q[~mask].T])
        self.qlenz = _np.max(q[~mask][:, 0]) + 1  # unpadded
        self.q = q
        self.mask = _np.copy(mask)
        self.accum = None
        self._tmp = None
        assemblenorm = correlator_tiles._getnorm(q, mask)
        with _np.errstate(divide='ignore'):
            self.invmean = 1 / (mean * assemblenorm)
        self.invmean[~_np.isfinite(self.invmean)] = 0
        self._N = 0
        self.finished = False
        self._qlen = qlen

    def suspend(self):
        """
        free tmp buffer
        """
        self._tmp = None

    def __exit__(self, *args):
        self.suspend()

    def __enter__(self):
        if self._tmp is None:
            tmp = alignedarray((self._qlen[0], self._qlen[1], self._qlen[2] + 2), zero=True)
            self._tmp = tmp
        if self.accum is None:
            accum = _np.zeros((self.qlenz, self._qlen[1], self._qlen[2] + 2))
            _np.subtract(accum, 0, out=accum)  # force allocation
            self.accum = accum
        return self

    def add(self, data):
        """
        does correlation of data and adds to internal accumulator
        data: array of shape T,P with T: non overlapping tiles, P pixel per Tile
        """
        d = atleastnd(data, 2)
        if not d.shape == self.q.shape[:-1]:
            print(d.shape, self.q.shape, flush=True)
            raise ValueError('shape missmatch')
        if self.finished:
            raise GeneratorExit('already finished')
        if self._tmp is None:
            tmp = alignedarray((self._qlen[0], self._qlen[1], self._qlen[2] + 2), zero=True)
            self._tmp = tmp
        else:
            _zero(self._tmp)
        if self.accum is None:
            accum = _np.zeros((self.qlenz, self._qlen[1], self._qlen[2] + 2))
            _np.subtract(accum, 0, out=accum)  # force allocation
            self.accum = accum
        d = d * self.invmean
        d[self.mask] = 0
        _addat(self._tmp, self.q, d)
        err = autocorrelate3(self._tmp)
        if err:
            raise RuntimeError(f'cython autocorrelations failed with error code {err}')
        _np.add(self.accum, self._tmp[: self.qlenz, ...], out=self.accum)
        self._N += 1

    def result(self, finish=False):
        """
        returns result of accumulated correlations
        finish: free accumulator and buffer.
        """
        if self.accum is None:
            return None

        _zero(self._tmp)
        assemblenorm = correlator_tiles._getnorm(self.q, self.mask)
        _addat(self._tmp, self.q, _np.sqrt(self.N) * _np.array(~self.mask, dtype=_np.float64) / assemblenorm)
        err = autocorrelate3(self._tmp)
        if err:
            raise RuntimeError(f'cython autocorrelations failed with error code {err}')

        res = _ne.evaluate(
            'where((norm<(10*N)), nan,accum/norm)',
            local_dict={'N': self._N, 'norm': self._tmp[: self.qlenz, ...], 'nan': _np.nan, 'accum': self.accum},
        )
        if finish:
            self.finished = True
            self.accum = None
            self._tmp = None
        res = unwrap(res)
        return res

    @property
    def shape(self):
        """
        shape of the result
        """
        return self.accum.shape

    @property
    def N(self):
        """
        number of images added
        """
        return self._N

    @staticmethod
    def _getnorm(q, mask):
        """
        returns amount of pixels with same q
        """
        maxq = _np.max(q.reshape(-1, 3), axis=0)
        hist = _np.histogramdd(q.reshape(-1, 3), bins=maxq + 1, range=[[-0.5, mq + 0.5] for mq in maxq], weights=~mask.ravel())[0]
        ret = hist[q.reshape(-1, 3)[:, 0], q.reshape(-1, 3)[:, 1], q.reshape(-1, 3)[:, 2]].reshape(q.shape[:2])
        ret[mask] = 1
        return ret


@_numba.njit(parallel=True)
def _addat(array, ind, input):
    """
    sets array to input at ind, parallel
    """
    nproc = ind.shape[0]
    nel = ind.shape[1]
    for i in _numba.prange(nproc):
        tinp = input[i, ...]
        tind = ind[i, ...]
        for j in range(nel):
            array[tind[j, 0], tind[j, 1], tind[j, 2]] += tinp[j]
    return


@_numba.njit(parallel=True)
def _zero(array):
    """
    set array to zero
    """
    a = array.ravel()
    for i in _numba.prange(len(a)):
        a[i] = 0
    return True


class correlator:
    def __init__(self, maskout, z):
        """
        3d fft correlator
        maskout: points on detector that will not be used, shape should be same as images to correlate
        resolution is fixed at dq=1
        """
        y, x = _np.meshgrid(_np.arange(maskout.shape[1], dtype=_np.float64), _np.arange(maskout.shape[0], dtype=_np.float64))
        x -= maskout.shape[0] / 2.0
        y -= maskout.shape[1] / 2.0
        d = _np.sqrt(x ** 2 + y ** 2 + z ** 2)
        qs = _np.array([(k / d * z) for k in (z, y, x)])
        qs = _np.rint(qs - _np.min(qs, (-1, -2), keepdims=True)).astype(int, copy=False)
        qlen = [fastlen(k) for k in 2 * (_np.max(qs, (-1, -2)) + 1)]
        maxq = _np.max(qs, (-1, -2))
        hist = _np.histogramdd(qs.reshape(3, -1).T, bins=maxq + 1, range=[[-0.5, mq + 0.5] for mq in maxq], weights=~maskout.ravel())[0]
        count = hist[qs[0].ravel(), qs[1].ravel(), qs[2].ravel()].reshape(maskout.shape)
        count[maskout] = _np.inf
        self._countcorrection = 1 / count
        self._mask = maskout
        self._qlen = qlen
        self._qs = qs
        self._tmp = None

    def _corr(self, input, maxqz):
        for image in input:
            _zero(self._tmp)
            _np.add.at(self._tmp, (*self._qs,), image * self._countcorrection)
            err = autocorrelate3(self._tmp)
            if err:
                raise RuntimeError(f"cython autocorrelations failed with error code {err}")
            yield self._tmp[: min(self._tmp.shape[0] // 2, maxqz), ...]

    @property
    def mask(self):
        """
        mask being used
        """
        return self._mask

    def corr(self, input, maxqz=_np.inf):
        """
        correlate one or multiple images
        return view that will be destroyed on next call, should be copied!
        """
        if self._tmp is None:
            self._tmp = alignedarray((self._qlen[0], self._qlen[1], self._qlen[2] + 2), zero=True)
        if isinstance(input, _np.ndarray):
            return next(self._corr((input,), maxqz))
        else:
            return self._corr(input, maxqz)

    def __enter__(self):
        if self._tmp is None:
            self._tmp = alignedarray((self._qlen[0], self._qlen[1], self._qlen[2] + 2), zero=True)
        return self

    def __exit__(self, *args):
        self.suspend()

    def suspend(self):
        self._tmp = None

    @staticmethod
    def unwrap(img):
        """
        unwraps a single correlation result
        """
        return _np.roll(img[..., :-2], shift=(img.shape[1] // 2, (img.shape[2] - 2) // 2), axis=(1, 2))
