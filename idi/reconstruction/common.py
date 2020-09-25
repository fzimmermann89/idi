import numba as _numba
import numpy as _np


@_numba.njit(nogil=True)
def _getidx(p, pmax, idmax):
    '''
    gives ids (up to idxmax) for job p of pmax jobs to do. will pair low ids with high ids for the same job
    '''
    idsperP = idmax // (2 * pmax)
    n = 2 * idsperP
    missing = idmax - (idsperP * (2 * pmax))
    if p < missing:
        if p < missing - pmax:
            n = n + 2
            out = _np.empty(n, dtype=_numba.int64)
            out[-1] = idmax - p - pmax - 1
            out[-2] = idmax - p - 1
        else:
            n = n + 1
            out = _np.empty(n, dtype=_numba.int64)
            out[-1] = idmax - p - 1
    else:
        out = _np.empty(n, dtype=_numba.int64)
    for (m, x) in enumerate(range(p * idsperP, (p + 1) * idsperP)):
        out[m] = x
    for (k, x) in enumerate(range(((2 * pmax - 1) - p) * idsperP, ((2 * pmax) - p) * idsperP)):
        out[m + k + 1] = x
    return out


def split(N, sections, return_lists=False):
    each, remainder = divmod(N, sections)
    splits = np.array(([0] + remainder * [each+1] + (sections-remainder) * [each])).cumsum()
    ranges = [range(splits[i],splits[i+1]) for i in range(sections)]
    if return_lists:
        return [list(i) for i in ranges]
    else:
        return ranges

def getidx_ordered(p, pmax, idmax):
    '''
    gives ids (up to idxmax) for job p of pmax jobs to do.
    '''
    return [i for i in range(p,idmax,pmax)]