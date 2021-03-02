import numpy as _np
import numba as _numba


def get_kernel(Natoms, Ndet, pixelsize, detz, k, nodist=True, nf=False):
    """
    returns a cpu implementation of the wavefield, used internally
    Natoms: Number of atoms
    Ndet: detector pixels
    pixelsize: detector pixelsize
    detz: detector distance
    k: angular wavenumber
    returns a function with signature complex64(:,:)(out complex64(:,:), atompositionsandphases float64[:,4])
        that will write the wavefield into out
    """
    maxx, maxy = int(Ndet[0]), int(Ndet[1])
    k = float(k)
    pixelsize = float(pixelsize)
    detz = float(detz)
    Natoms = int(Natoms)

    @_numba.njit(inline='always', fastmath=True)
    def nfphase(atom, detx, dety):
        dist = _np.sqrt((detx - atom[0]) ** 2 + (dety - atom[1]) ** 2 + (detz - atom[2]) ** 2)
        phase = (dist - detz + atom[2]) * k + atom[3]
        if nodist:
            px = _np.cos(phase)
            py = _np.sin(phase)
        else:
            rdist = 1 / dist
            px = rdist * _np.cos(phase)
            py = rdist * _np.sin(phase)
        return px, py

    @_numba.njit(inline='always', fastmath=True)
    def ffphase(atom, qx, qy, qz, rdist):
        phase = -(atom[0] * qx + atom[1] * qy + atom[2] * qz) + atom[3]
        if nodist:
            px = _np.cos(phase)
            py = _np.sin(phase)
        else:
            px = rdist * _np.cos(phase)
            py = rdist * _np.sin(phase)
        return px, py

    @_numba.njit('complex128[:, ::1],float64[:, ::1]', parallel=True, fastmath={'contract', 'arcp', 'ninf', 'nnan'})
    def kernel(ret, atom):
        for x in _numba.prange(maxx):
            for y in range(maxy):
                detx = (x - maxx / 2) * pixelsize
                dety = (y - maxy / 2) * pixelsize
                accumx = 0.0
                accumy = 0.0
                cx = 0.0
                cy = 0.0
                if not nf:
                    rdist = 1 / _np.linalg.norm(_np.array([detx, dety, detz]))
                    qz = k * (detz * rdist - 1)
                    qx = k * (detx * rdist)
                    qy = k * (dety * rdist)

                for i in range(Natoms):
                    if nf:
                        px, py = nfphase(atom[i, :], detx, dety)
                    else:
                        px, py = ffphase(atom[i, :], qx, qy, qz, rdist)

                    # kahan summation
                    yx = px - cx
                    yy = py - cy
                    tx = accumx + yx
                    ty = accumy + yy
                    cx = (tx - accumx) - yx
                    cy = (ty - accumy) - yy
                    accumx = tx
                    accumy = ty

                ret[x, y] = accumx + 1j * accumy

    return kernel


def simulate(Nimg, simobject, Ndet, pixelsize, detz, k, settings='', verbose=False, *args, **kwargs):
    """
    returns an array of simulated wavefields
    parameters:
    Nimg: number of wavefields to simulate
    simobject: a simobject whose get() returns an Nx4 array with atoms in the first
        and (x,y,z,phase) of each atom in the last dimension
    Ndet: pixels on the detector
    pixelsize: size of one pixel in same unit as simobjects unit (usually um)
    detz: detector distance in same unit as simobjects unit (usually um)
    k: angular wavenumber
    settings: string
        if it contains 'scale', 1/r  scaling is performed
        if it contains 'nf', no far field approximation is made
    """

    if _np.size(Ndet) == 1:
        Ndet = [Ndet, Ndet]
    result = _np.empty((Nimg, Ndet[0], Ndet[1]), dtype=_np.complex128)
    gen = simulate_gen(simobject, Ndet, pixelsize, detz, k, settings)
    for n in range(0, Nimg):
        if verbose:
            print(n, end='', flush=True)
        result[n] = next(gen)
        if verbose:
            print('. ', end='', flush=True)
    return result


def simulate_gen(simobject, Ndet, pixelsize, detz, k, settings='', *args, **kwargs):
    """
    returns a generator that yields simulated wavefields
    parameters:
    simobject: a simobject whose get() returns an Nx4 array with atoms in the first
        and (x,y,z,phase) of each atom in the last dimension
    Ndet: pixels on the detector
    pixelsize: size of one pixel in same unit as simobjects unit (usually um)
    detz: detector distance in same unit as simobjects unit (usually um)
    k: angular wavenumber
    settings: string
        if it contains 'scale', 1/r  scaling is performed
        if it contains 'nf', no far field approximation is made
    """

    nodist = 'scale' not in settings
    nf = 'nf' in settings

    if _np.size(Ndet) == 1:
        Ndet = [Ndet, Ndet]

    result = _np.empty((Ndet[0], Ndet[1]), dtype=_np.complex128)
    f = get_kernel(simobject.N, Ndet, pixelsize, detz, k, nodist=nodist, nf=nf)
    while True:
        atoms = simobject.get()
        f(result, atoms)
        yield result
