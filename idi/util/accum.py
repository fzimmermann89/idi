from __future__ import division as _future_division, print_function as _future_print
import numba as _numba
import numpy as _np
import scipy.ndimage as _snd
import numexpr as _ne

class accumulator:
    def __init__(self, like=None):
        self._n = 0
        if like is None:
            self._mean = None
            self._nvar = None
        else:
            self._mean = _np.zeros_like(like)
            self._nvar = _np.zeros_like(like)

    def __repr__(self):
        print(type(self._mean))
        return 'accumulator[%i]' % self._n
    
    def add(self, value, count=1):
        self._n += count
        if self._mean is None:
            self._mean = _np.asarray(value).astype(_np.float64)
            self._nvar = _np.zeros_like(value).astype(_np.float64)
        else:
            with _np.errstate(divide='ignore', invalid='ignore'):
                delta = value - self._mean
                self._mean = _np.add(self._mean, delta / self._n, where=(count!=0), out=self._mean)
                self._nvar = _ne.evaluate(
                    'nvar + delta * (value - mean)',
                    local_dict={'nvar': self._nvar, 'value': value, 'delta': delta, 'mean': self._mean},
                )

    def __len__(self):
        return _np.max(self._n)
    
    @property
    def shape(self):
        if self._mean is not None:
            return _np.asarray(self._mean).shape
        else:
            return 0
    @property
    def mean(self):
        if self._mean is not None:
            return self._mean
        else:
            return 0

    @property
    def var(self):
        if self._nvar is not None:
            return self._nvar / self._n
        else: 
            return 0

    @property
    def std(self):
        return _np.sqrt(self.var)

    
