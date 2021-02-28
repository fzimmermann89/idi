import numpy as _np
import numexpr as _ne


class accumulator:
    def __init__(self, maxmin=False):
        """
        Accumulate nd arrays and calculate mean and variance
        :param maxmin: Keep track of maximum and minimum
        """
        self._n = _np.array(0)
        self._mean = None
        self._nvar = None
        self._maxmin = maxmin
        self._result = None

    def __repr__(self):
        return 'accumulator[%i]' % self._n

    #  basic idea:
    #  count += 1
    #  delta = newValue - mean
    #  mean += delta / count
    #  delta2 = newValue - mean
    #  M2 += delta * delta2

    def add(self, value, count=1.0):
        self._result = None
        count = _np.array(count, value.dtype)
        if self._mean is None:
            with _np.errstate(divide='ignore', invalid='ignore'):
                self._mean = value / count
                self._mean[~_np.isfinite(self._mean)] = 0
            self._nvar = _np.zeros_like(value)
            self._n = _np.array(count, self._mean.dtype)
            if self._maxmin is True:
                self._max = _np.copy(value)
                self._min = _np.copy(value)
        else:
            tmp = _ne.evaluate('value/count - mean', local_dict={'count': count, 'value': value, 'mean': self._mean})
            with _np.errstate(divide='ignore', invalid='ignore'):
                _ne.evaluate('where(count>0,n+1,n)', local_dict={'count': count, 'n': self._n}, out=self._n)
                _ne.evaluate(
                    'where(count>0,mean+tmp/n,mean)',
                    local_dict={'value': value, 'tmp': tmp, 'mean': self._mean, 'count': count, 'n': self._n},
                    out=self._mean,
                )
                _ne.evaluate(
                    'where(count>0, nvar + tmp * (value/count - mean), nvar)',
                    local_dict={'count': count, 'value': value, 'tmp': tmp, 'mean': self._mean, 'nvar': self._nvar},
                    out=self._nvar,
                )
            if self._maxmin is True:
                if isinstance(count, _np.ndarray) and len(count.shape) > 0:
                    self._max[count > 0] = _np.maximum(self._max, value)[count > 0]
                    self._min[count > 0] = _np.minimum(self._min, value)[count > 0]
                else:
                    _np.maximum(self._max, value, out=self._max)
                    _np.minimum(self._min, value, out=self._min)

    def __len__(self):
        return _np.max(self._n)

    @property
    def mean(self, copy=False):
        if self._n.size > 1 or copy:
            mean = _np.copy(self._mean)
            mean[self.n == 0] = _np.nan
            return mean
        else:
            return self._mean

    @property
    def n(self):
        return self._n

    @property
    def var(self):
        if _np.any(self._n>0):
            with _np.errstate(all='ignore'):
                return self._nvar / self._n
        else:
            return None

    @property
    def std(self):
        if _np.any(self._n > 0):
            with _np.errstate(all='ignore'):
                return _np.sqrt(self.var)
        else:
            return None

    @property
    def max(self):
        if self._maxmin is False:
            raise NotImplementedError('set maxmin to True when creating accumulator to keep min and max')
        return self._max

    @property
    def min(self):
        if self._maxmin is False:
            raise NotImplementedError('set maxmin to True when creating accumulator to keep min and max')
        return self._min

    @property
    def result(self):
        if self._result is None:
            self._result = _frozenaccumulator(self)
        return self._result


class _frozenaccumulator:
    """
    Accumulator that doesn't accumulate anymore
    """

    def __init__(self, accum):
        self._mean = _np.copy(accum.mean)
        self._std = _np.copy(accum.std)
        self._n = _np.copy(accum.n)
        self._invmean = _np.nan_to_num(1.0 / self._mean)
        self._invstd = _np.nan_to_num(1.0 / self._std)

    @property
    def mean(self):
        return self._mean

    @property
    def std(self):
        return self._std

    @property
    def invstd(self):
        return self._invstd

    @property
    def invmean(self):
        return self._invmean

    @property
    def n(self):
        return self._n
