#!/bin/env python
from __future__ import division as _future_div ,print_function as _future_print
import numpy as _np
import numba as _numba
from six.moves import range #for python2 compatibility
from .common import _getidx
pmax=12

def corr(input,z):
    _numba.config.NUMBA_NUM_THREADS=pmax
    """
    radial profile of correlation
    for one NxN array or nx(NxN) arrays
    """
    if input.ndim==2:
        return _pradcorr(input,z)
    elif input.ndim==3:
        return _np.sum(_pradcorrs(input,z),axis=0)
    else:
        raise TypeError

@_numba.njit(parallel=False)
def _radcorr(input,z):
    """
    radial profile of correlation
    for one NxN array
    """
    N=max(input.shape)
    xi,yi=(_np.where(input>0))
    Nhits=len(xi)
    qlen=int(_np.ceil(N*2))
    tmp=_np.zeros(qlen,dtype=_np.uint64)
    x=(xi).astype(_numba.float64)-N/2.
    y=(yi).astype(_numba.float64)-N/2.
    d=_np.sqrt(x**2+y**2+z**2)
    kx=x/d*z
    ky=y/d*z
    kz=z/d*z
    
    for n in range(Nhits):
        for m in range(n):
            qz=kz[n]-kz[m]
            qx=kx[n]-kx[m]
            qy=ky[n]-ky[m]
            q=int(_np.rint(_np.sqrt(qx**2+qy**2+qz**2)))
            if q>qlen:
                 #print(q,qlen,(qx,qy,qx))
                pass
            else:
                tmp[q]+=input[xi[n],yi[n]]*input[xi[m],yi[m]]
    return tmp        
        
               
@_numba.njit(parallel=True,cache=False,nogil=True,fastmath=True)
def _pradcorr(input,z):
    """
    radial profile of correlation
    for one NxN array (parallel version)
    """
    #TODO worksize aka getidx
    print(pmax)
    N=max(input.shape)   
    xi,yi=(_np.where(input))
    x=(xi).astype(_numba.float64)-N/2.
    y=(yi).astype(_numba.float64)-N/2.
    Nhits=len(xi)
    qlen=int(_np.ceil(N*2))
    tmp=_np.zeros((pmax,qlen),dtype=_numba.uint64)
    d=_np.sqrt(x**2+y**2+z**2)
    kx=(x/d)*z
    ky=(y/d)*z
    kz=z/d*z
    for p in _numba.prange(pmax):
        idx=_getidx(p,pmax,Nhits)
        for n in idx:
            for m in range(n):
                qz=kz[n]-kz[m]
                qx=kx[n]-kx[m]
                qy=ky[n]-ky[m]
                q=int(_np.rint(_np.sqrt(qx**2+qy**2+qz**2)) )
                if q>qlen:
                     #print(q,qlen,(qx,qy,qx))
                    pass
                else:
                    tmp[p,q]+=input[xi[n],yi[n]]*input[xi[m],yi[m]]
        print(p)
    out=_np.zeros(qlen)
    for n in range(pmax):
        out+=tmp[n,...]
    return out 




@_numba.njit(parallel=True)
def _pradcorrs(input,z):
    """
    radial profile of correlation
    for n NxN arrays, parallel over n
    """
    qlen=int(_np.ceil(2*max(input.shape[-2:])))
    out=_np.zeros((input.shape[0],qlen),dtype=_numba.uint64)
    for n in _numba.prange(input.shape[0]):
        out[n]=(_radcorr(input[n,...],z))
    return out