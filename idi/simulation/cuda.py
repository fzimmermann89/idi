import numpy as _np
import cupy as _cp
from pathlib import Path as _Path
from ..util import rotation as _rotation

code = (_Path(__file__).parent / 'sim.cu').read_text()


def _pinned(shape, dtype):
    size = int(_np.prod(shape))
    mem = _cp.cuda.alloc_pinned_memory(size * _np.dtype(dtype).itemsize)
    ret = _np.frombuffer(mem, dtype, size).reshape(shape, order='C')
    return ret


def simulate_gen(simobject, Ndet, pixelsize, detz, k=None, settings="double", maximg=_np.inf, detangles=(0, 0), *args, **kwargs):
    """
    returns an array of simulated wavefields
    parameters:
    simobject: a simobject whose get() returns an Nx4 array with atoms in the first and (x,y,z,phase) of each atom in the last dimension
    Ndet: pixels on the detector
    pixelsize: size of one pixel in same unit as simobjects unit (usally um)
    detz: detector distance in same unit as simobjects unit (usally um)
    k: angular wavenumber, if None (default) get from simobject
    settings: string, default: double_ff_nodist can contain
        single: use single precision (else: double precission)
        nf:     use near field
        scale: apply 1/r intensity scaling
        secondorder: use second order in far field approximation (Fresnel)
        nofast: no fast math
        unknown options will be silently ignored.
    maximg: generate StopIteration after maximg images.
    detangles: (theta,phi) angles of detector rotation around origin
    """
    if k is None:
        k = simobject.k
    if "single" in settings:
        intype, outtype, kernelname = _np.float32, _np.complex64, "wfkernelf"
    else:
        intype, outtype, kernelname = _np.float64, _np.complex128, "wfkerneld"
    options = []
    if "scale" not in settings:
        options += ["-Dnodist"]
    if "nf" in settings:
        options += ["-Dusenf"]
    if "secondorder" in settings:
        options += ["-Dsecondorder"]
    if "nofast" not in settings:
        options += ["--use_fast_math"]

    if _np.size(Ndet) == 1:
        Ndet = [Ndet, Ndet]

    if _np.any(detangles):
        M = _cp.array(_rotation(*detangles, 0), _np.float64)
    else:
        M = _cp.eye(3)
        options += ["-Dnodetrot"]

    threadsperblock = (16, 16, 1)
    blockspergrid_x = int(_np.ceil(Ndet[0] / threadsperblock[0]))
    blockspergrid_y = int(_np.ceil(Ndet[1] / threadsperblock[1]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    module = _cp.RawModule(code=code, backend="nvcc", options=tuple(["--std=c++11", "-O3", "--restrict"] + options))
    kernel = module.get_function(kernelname)

    d_wf = _cp.zeros((Ndet[0], Ndet[1]), dtype=outtype, order='C')
    d_atoms = _cp.zeros((simobject.N, 4), intype, order='C')
    h_atoms = _pinned((simobject.N, 4), intype)

    def _gen():
        h_atoms[:, :3], h_atoms[:, 3:] = simobject.get2()
        d_atoms.set(h_atoms)
        kernel(
            blockspergrid, threadsperblock, (d_wf, d_atoms, M, float(detz), float(pixelsize), float(k), int(Ndet[0]), int(Ndet[1]), int(simobject.N))
        )
        count = 1
        while True:
            if count == maximg:
                yield d_wf.get()
            elif count > maximg:
                return
            else:
                h_atoms[:, :3], h_atoms[:, 3:] = simobject.get2()
                d_atoms.set(h_atoms)
                ret = d_wf.get()
                kernel(
                    blockspergrid,
                    threadsperblock,
                    (d_wf, d_atoms, M, float(detz), float(pixelsize), float(k), int(Ndet[0]), int(Ndet[1]), int(simobject.N)),
                )
                yield ret
            count += 1

    return _gen()


def simulate(Nimg, simobject, Ndet, pixelsize, detz, k=None, settings="double", verbose=True, detangles=(0, 0), *args, **kwargs):
    """
    returns an array of simulated wavefields
    parameters:
    Nimg: number of wavefields to simulate
    simobject: a simobject whose get() returns an Nx4 array with atoms in the first and (x,y,z,phase) of each atom in the last dimension
    Ndet: pixels on the detector
    pixelsize: size of one pixel in same unit as simobjects unit (usally um)
    detz: detector distance in same unit as simobjects unit (usally um)
    k: angular wavenumber
    settings: string, default: double_ff_nodist can contain
        single: use single precision
        nf:     use near field
        scale: apply 1/r intensity scaling
        secondorder: use second order in far field approximation (Fresnel)
        nofast: no fast math
        unknown options will be silently ignored.
    detangles: (theta,phi) angles of detector rotation around origin
    """
    if _np.size(Ndet) == 1:
        Ndet = [Ndet, Ndet]
    gen = simulate_gen(simobject, Ndet, pixelsize, detz, k, settings=settings, maximg=Nimg, detangles=detangles)
    result = _np.empty((Nimg, Ndet[0], Ndet[1]), _np.complex128)
    try:
        for i, img in enumerate(gen):
            if verbose:
                print(i, end='. ', flush=True)
            result[i, ...] = img
        return result
    except KeyboardInterrupt:
        if verbose:
            print('Interrupted')
        return result[:i]
