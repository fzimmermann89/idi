from __future__ import division as _future_div, print_function as _future_print
from six import print_ as _print
import numpy as _np
import numba as _numba
import IPython


def wavefield_kernel(Natoms, Ndet, pixelsize, detz, k):
    '''
    returns a cpu implementation of the wavefield, used internally
    Natoms: Number of atoms
    Ndet: detector pixels
    pixelsize: detector pixelsize
    detz: detector distance
    k: angular wavenumber
    returns a function with signature complex64(:,:)(out complex64(:,:), atompositionsandphases float64[:,4]) 
        that will write the wavefield into out 
    '''
    maxx, maxy = Ndet

    @_numba.njit('complex64[:, ::1],float64[:, ::1]', parallel=True, fastmath=True)
    def wfkernel(ret, atom):
        '''
        '''
        for x in _numba.prange(maxx):
            for y in range(maxy):
                detx = (x - maxx / 2) * pixelsize
                dety = (y - maxy / 2) * pixelsize
                wfx = 0
                wfy = 0
                for i in range(Natoms):
                    dist = _np.sqrt((detx - atom[i, 0]) ** 2 + (dety - atom[i, 1]) ** 2 + (detz - atom[i, 2]) ** 2)
                    rdist = 1 / dist
                    phase = (dist * k) % (2 * _np.pi) + atom[i, 3]
                    real = _np.cos(phase)
                    imag = _np.sin(phase)
                    wfx += real * rdist
                    wfy += imag * rdist

                ret[x, y] = wfx + 1j * wfy

    return wfkernel


def simulate(Nimg, simobject, Ndet, pixelsize, detz, k, verbose=True):
    '''
    returns an array of simulated wavefields
    parameters:
    Nimg: number of wavefields to simulate
    simobject: a simobject whose get() returns an Nx4 array with atoms in the first and (x,y,z,phase) of each atom in the last dimension
    Ndet: pixels on the detector
    pixelsize: size of one pixel in same unit as simobjects unit (usally um)
    detz: detector distance in same unit as simobjects unit (usally um)
    k: angular wavenumber
    '''
    if _np.size(Ndet) == 1:
        Ndet = [Ndet, Ndet]
    result = _np.empty((Nimg, Ndet[0], Ndet[1]), dtype=complex)
    h_wf1 = _np.empty((Ndet[0], Ndet[1]), dtype=_np.complex64)
    fwavefield = wavefield_kernel(simobject.N, Ndet, pixelsize, detz, k)
    for n in range(0, Nimg):
        if verbose:
            _print(n, end='', flush=True)
        h_atoms1 = simobject.get()
        if verbose:
            _print('.', end='', flush=True)
        fwavefield(h_wf1, h_atoms1)
        result[n, ...] = h_wf1.copy()
        if verbose:
            _print('. ', end='', flush=True)
    return result


def simulate_gen(simobject, Ndet, pixelsize, detz, k):
    '''
    returns a generator that yields simulated wavefields
    parameters:
    simobject: a simobject whose get() returns an Nx4 array with atoms in the first and (x,y,z,phase) of each atom in the last dimension
    Ndet: pixels on the detector
    pixelsize: size of one pixel in same unit as simobjects unit (usally um)
    detz: detector distance in same unit as simobjects unit (usally um)
    k: angular wavenumber
    '''
    if _np.size(Ndet) == 1:
        Ndet = [Ndet, Ndet]
    h_wf1 = _np.empty((Ndet[0], Ndet[1]), dtype=_np.complex64)
    fwavefield = wavefield_kernel(simobject.N, Ndet, pixelsize, detz, k)
    while True:
        h_atoms1 = simobject.get()
        fwavefield(h_wf1, h_atoms1)
        yield h_wf1.copy()