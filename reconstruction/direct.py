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
pmax=8
#Todo qlenz

@numba.njit(parallel=False)
def _corr(input,z):
    N=max(input.shape)
    xi,yi=(np.where(input>0))
    Nhits=len(xi)
    qlenx=int(N)
    qleny=int(N)
    qlenz=int(np.ceil(z-np.sqrt(z**2-N**2/4)))
    tmp=np.zeros((qlenx,qleny,qlenz),dtype=np.uint64)
    x=(xi).astype(numba.float64)-N/2.
    y=(yi).astype(numba.float64)-N/2.
    d=np.sqrt(x**2+y**2+z**2)
    kx=x/d*z
    ky=y/d*z
    kz=z/d*z

    for n in range(Nhits):
        for m in range(n):
            qz=int(np.rint(np.abs(kz[n]-kz[m])))
            qx=int(np.rint(np.abs(kx[n]-kx[m])))
            qy=int(np.rint(np.abs(ky[n]-ky[m])))
            if qx>=qlenx or qy>=qleny or qz>=qlenz:
                #print((qx,qy,qx))
                pass
            else:
                tmp[qx,qy,qz]+=input[xi[n],yi[n]]*input[xi[m],yi[m]]
    return tmp

@numba.njit(nogil=True,parallel=True,fastmath=True)
def _pcorr(input,z):
    N=max(input.shape)
    xi,yi=(np.where(input>0))
    Nhits=len(xi)
    qlenx=int(N)
    qleny=int(N)
    qlenz=int(np.ceil(z-np.sqrt(z**2-N**2/4)))
    tmp=np.zeros((pmax,qlenx,qleny,qlenz),dtype=np.uint64)
    x=(xi).astype(numba.float64)-N/2.
    y=(yi).astype(numba.float64)-N/2.
    d=np.sqrt(x**2+y**2+z**2)
    kx=x/d*z
    ky=y/d*z
    kz=z/d*z
    for p in numba.prange(pmax):
        idx=_getidx(p,pmax,Nhits)
        for n in idx:
            for m in range(n):
                qz=int(np.rint(np.abs(kz[n]-kz[m])))
                qx=int(np.rint(np.abs(kx[n]-kx[m])))
                qy=int(np.rint(np.abs(ky[n]-ky[m])))
                if qx>=qlenx or qy>=qleny or qz>=qlenz:
                    pass
                    #print((qx,qy,qx))
                else:
                    tmp[p,qx,qy,qz]+=input[xi[n],yi[n]]*input[xi[m],yi[m]]
    out=np.zeros((qlenx,qleny,qlenz))
    for p in range(pmax):
        out+=tmp[p,...]
    return out 

@numba.njit(nogil=True)
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


def corr(input,z):
    """
    correlation
    for one NxN array or nx(NxN) arrays
    """
    numba.config.NUMBA_NUM_THREADS=pmax
    if input.ndim==2:
        return _pcorr(input,z)
    elif input.ndim==3:
        return np.sum(_pcorrs(input,z),axis=0)
    else:
        raise TypeError

@numba.njit(parallel=True)
def _pcorrs(input,z):
    N=max(input.shape[-2:])
    Nimg=input.shape[0]
    qlenx=int(N)
    qleny=int(N)
    qlenz=int(np.ceil(z-np.sqrt(z**2-N**2/4)))
    Nperp=int(np.floor(Nimg/pmax))
    out=np.zeros((pmax,qlenx,qleny,qlenz),dtype=np.uint64)
    for p in numba.prange(pmax):
        for n in range(p*Nperp,(p+1)*Nperp):
            out[p]+=(_corr(input[n,...],z))
    return out