from __future__ import division, print_function
import numpy as _np
from multiprocessing import Pool as _mpPool

# rebin
def rebin(arr, n):
    shape = (arr.shape[0], arr.shape[1] // (2 ** n), (2 ** n), arr.shape[2] // (2 ** n), (2 ** n))
    return arr.reshape(shape).mean(-1).mean(-2)


# photon samling
def _randomphotons(args):
    def _randomphotons_unpacked(prob, Nphotons):
        photons = _np.zeros_like(prob, dtype=_np.uint32)
        hits = _np.random.choice(range(prob.size), size=Nphotons, p=prob.flatten())
        _np.add.at(photons.ravel(), hits, 1)
        return photons

    return _randomphotons_unpacked(*args)


def randomphotons(incoherent, Nphotons, pmax=8):
    gesamt = _np.sum(incoherent, axis=(1, 2))
    probs = incoherent / gesamt[:, None, None]
    Ns = (Nphotons * gesamt / _np.mean(gesamt)).astype(int)
    args = zip(probs, Ns)
    pool = _mpPool(processes=pmax)
    out = pool.map(_randomphotons, args)
    pool.close()
    pool.join()
    return out
