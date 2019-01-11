#!/bin/env python
from __future__ import division,print_function

import numpy as np
import math
import numexpr as ne
from numpy import pi
import numba
from optparse import OptionParser
from multiprocessing import Pool
import functools
from six.moves import range #for python2 compatibility
pmax=12

def corr(input,z):
    numba.config.NUMBA_NUM_THREADS=pmax
    """
    radial profile of correlation
    for one NxN array or nx(NxN) arrays
    """
    if input.ndim==2:
        return _pradcorr(input,z)
    elif input.ndim==3:
        return np.sum(_pradcorrs(input,z),axis=0)
    else:
        raise TypeError

@numba.njit(parallel=False)
def _radcorr(input,z):
    """
    radial profile of correlation
    for one NxN array
    """
    N=max(input.shape)
    xi,yi=(np.where(input>0))
    Nhits=len(xi)
    qlen=int(np.ceil(N*2))
    tmp=np.zeros(qlen,dtype=np.uint64)
    x=(xi).astype(numba.float64)-N/2.
    y=(yi).astype(numba.float64)-N/2.
    d=np.sqrt(x**2+y**2+z**2)
    kx=x/d*z
    ky=y/d*z
    kz=z/d*z
    
    for n in range(Nhits):
        for m in range(n):
            qz=kz[n]-kz[m]
            qx=kx[n]-kx[m]
            qy=ky[n]-ky[m]
            q=int(np.rint(np.sqrt(qx**2+qy**2+qz**2)))
            if q>qlen:
                 #print(q,qlen,(qx,qy,qx))
                pass
            else:
                tmp[q]+=input[xi[n],yi[n]]*input[xi[m],yi[m]]
    return tmp        
        
               
@numba.njit(parallel=True,cache=False,nogil=True,fastmath=True)
def _pradcorr(input,z):
    """
    radial profile of correlation
    for one NxN array (parallel version)
    """
    #TODO worksize aka getidx
    print(pmax)
    N=max(input.shape)   
    xi,yi=(np.where(input))
    x=(xi).astype(numba.float64)-N/2.
    y=(yi).astype(numba.float64)-N/2.
    Nhits=len(xi)
    qlen=int(np.ceil(N*2))
    tmp=np.zeros((pmax,qlen),dtype=numba.uint64)
    d=np.sqrt(x**2+y**2+z**2)
    kx=(x/d)*z
    ky=(y/d)*z
    kz=z/d*z
    for p in numba.prange(pmax):
        idx=_getidx(p,pmax,Nhits)
        for n in idx:
            for m in range(n):
                qz=kz[n]-kz[m]
                qx=kx[n]-kx[m]
                qy=ky[n]-ky[m]
                q=int(np.rint(np.sqrt(qx**2+qy**2+qz**2)) )
                if q>qlen:
                     #print(q,qlen,(qx,qy,qx))
                    pass
                else:
                    tmp[p,q]+=input[xi[n],yi[n]]*input[xi[m],yi[m]]
        print(p)
    out=np.zeros(qlen)
    for n in range(pmax):
        out+=tmp[n,...]
    return out 

@numba.njit(nogil=True,cache=False)
def _getidx(p,pmax,idmax):
    idsperP=idmax//(2*pmax)
    n=2*idsperP
    missing=idmax-(idsperP*(2*pmax))
    if p<missing:
        if p<missing-pmax:
            n=n+2
            out=np.empty(n,dtype=numba.int64)
            out[-1]=idmax-p-pmax-1
            out[-2]=idmax-p-1
        else:
            n=n+1
            out=np.empty(n,dtype=numba.int64)
            out[-1]=idmax-p-1
    else:
        out=np.empty(n,dtype=numba.int64)
    for (m,x) in enumerate(range(p*idsperP,(p+1)*idsperP)):
        out[m]=x
    for (k,x) in enumerate(range(((2*pmax-1)-p)*idsperP,((2*pmax)-p)*idsperP)):
        out[m+k+1]=x
    return out



@numba.njit(parallel=True)
def _pradcorrs(input,z):
    """
    radial profile of correlation
    for n NxN arrays, parallel over n
    """
    qlen=int(np.ceil(2*max(input.shape[-2:])))
    out=np.zeros((input.shape[0],qlen),dtype=numba.uint64)
    for n in numba.prange(input.shape[0]):
        out[n]=(_radcorr(input[n,...],z))
    return out
  