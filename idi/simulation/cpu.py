from __future__ import division as _future_div, print_function as _future_print
from six import print_ as _print
import numpy as _np
import numba as _numba
import IPython


def wavefield_kernel(Natoms, Ndet, pixelsize, detz,k):
    maxx = maxy = Ndet
    
    @_numba.njit("complex64[:, ::1],float64[:, ::1]",parallel=True,fastmath=True)
    def wfkernel(ret,atom):
        for x in _numba.prange(maxx):
            for y in range(maxy):
                detx = (x-maxx/2)*pixelsize
                dety = (y-maxy/2)*pixelsize
                wfx = 0;
                wfy = 0;
                for i in range(Natoms):
                    dist = _np.sqrt(
                          (detx-atom[0,i])**2 
                         +(dety-atom[1,i])**2
                         +(detz-atom[2,i])**2 
                    )
                    rdist = 1/dist
                    phase = (dist*k)%(2*_np.pi)+atom[3,i]
                    real = _np.cos(phase)
                    imag = _np.sin(phase)                
                    wfx += real*rdist
                    wfy += imag*rdist

                ret[x,y]= wfx +1j*wfy
    return wfkernel
           
def simulate(Nimg,simobject,Ndet,pixelsize,detz,k,verbose=True):
    result=_np.empty((Nimg,Ndet,Ndet),dtype=complex)
    h_wf1=_np.empty((Ndet,Ndet),dtype=_np.complex64)
    fwavefield = wavefield_kernel(simobject.N, Ndet, pixelsize, detz,k)
    for n in range(0, Nimg):
        if verbose: _print(n, end='',flush=True)
        h_atoms1 = simobject.get()
        if verbose: _print('.', end='',flush=True)
        fwavefield(h_wf1,h_atoms1)
        result[n,...]=h_wf1.copy()
        if verbose: _print('. ', end='',flush=True)
    return result