code = """
/*
*This should be 10 lines of thrust code, but it's not. 
*Partially because it started as pycuda, partially because it had to run on an old gpu. 
*Can run as double or single precision with e^iqr (Fraunhofer) approximation (scattering direction constant within sample), second order in x,y (Fresnel)
*or in double precision with e^ik(R-r) for the phase (correct for incoming plane wave)
*/
#include <math.h>
#include <stdbool.h>

#define posx (threadIdx.x + blockDim.x * blockIdx.x)
#define posy (threadIdx.y + blockDim.y * blockIdx.y)

#ifndef nodist
    #define nodist 0
#endif
#ifndef secondorder
    #define secondorder 0
#endif
#ifndef usenf
    #define phaseop phaseff
#else
    #define phaseop phasenf
#endif

template <typename T, int cn> struct Vec;
template <> struct Vec<float, 2> {typedef float2 type; static __device__ __host__ __forceinline__ type make(float a, float b) {return make_float2(a,b);}};
template <> struct Vec<double, 2> {typedef double2 type; static __device__ __host__ __forceinline__ type make(double a, double b) {return make_double2(a,b);}};
template <> struct Vec<float, 3> {typedef float3 type; static __device__ __host__ __forceinline__ type make(float a, float b, float c) {return make_float3(a,b,c);}};
template <> struct Vec<double, 3> {typedef double3 type; static __device__ __host__ __forceinline__ type make(double a, double b, double c) {return make_double3(a,b,c);}};
template <> struct Vec<float, 4> {typedef float4 type; static __device__ __host__ __forceinline__ type make(float a, float b, float c, float d) {return make_float4(a,b,c,d);}};
template <> struct Vec<double, 4> {typedef double4 type; static __device__ __host__ __forceinline__ type make(double a, double b, double c, double d) {return make_double4(a,b,c,d);}};

__device__ __forceinline__ void sc(float x, float * sptr, float * cptr) {
    sincosf(x, sptr, cptr);
}

__device__ __forceinline__ void sc(double x, double * sptr, double * cptr) {
    sincos(x, sptr, cptr);
}

template < typename Ti, typename To >
struct phasenf {
    double detx, dety, detz, k;
    typedef typename Vec < To, 2 > ::type To2;
    typedef typename Vec < Ti, 4 > ::type Ti4;

    __device__ __forceinline__ phasenf(double _detx, double _dety, double _detz, double _k) {
        detx = _detx;
        dety = _dety;
        detz = _detz;
        k = _k;
    }
    __device__ To2 operator()(const Ti4 & atom) const {
        double dist = norm3d(
            (detx - (double) atom.x),
            (dety - (double) atom.y),
            (detz - (double) atom.z)
        );

        double phase = (dist - detz + atom.z) * k + atom.w;
        double real, imag;
        sc(phase, & imag, & real);
        if (nodist) return Vec < To, 2 > ::make((To)(real), (To)(imag));
        To rdist = 1 / ((To) dist);
        return Vec < To, 2 > ::make(((To) real) * rdist, ((To) imag) * rdist);
    }
};

template < typename Ti, typename To >
struct phaseff {
    To qx, qy, qz, c, rdist;
    typedef typename Vec < To, 2 > ::type To2;
    typedef typename Vec < Ti, 4 > ::type Ti4;
    
    __device__ __forceinline__ phaseff(double _detx, double _dety, double _detz, double _k) {
        double rdistd = rnorm3d(_detx, _dety, _detz);
        qz = (To)(_k * (_detz * rdistd - 1));
        double kinvd = _k * rdistd;
        qx = (To)(_detx * kinvd);
        qy = (To)(_dety * kinvd);
        c = (To)(kinvd / 2);
        rdist = (To) rdistd;
    }
    __device__ __forceinline__ To2 operator()(const Ti4 & atom) const {
        To phase = -(atom.x * qx + atom.y * qy + atom.z * qz) + atom.w;
        if (secondorder) phase += c * (atom.x * atom.x + atom.y * atom.y); /* this is the second order in x and y, still first in detz*/
        To real, imag;
        sc(phase, & imag, & real);
        if (nodist) return Vec < To, 2 > ::make(real, imag);
        return Vec < To, 2 > ::make(rdist * real, rdist * imag);

    }
};

__device__ __forceinline__ double4 __ldg(const double4 * d4) {
    return *d4;
}

template < typename Ti, typename To >
struct wfkernel {
    typedef typename Vec < To, 2 > ::type To2;
    typedef typename Vec < Ti, 4 > ::type Ti4;
    __device__ __forceinline__ void operator()(To2 __restrict__ * ret, Ti4 const __restrict__ * atoms, double detz, double pixelsize, double k, int maxx, int maxy, int Natoms) const {
        int x = posx, y = posy;
        if ((x < maxx) && (y < maxy)) {
            double detx = (x - maxx / 2) * pixelsize;
            double dety = (y - maxy / 2) * pixelsize;
            To accumx = 0, accumy = 0, cx = 0, cy = 0;
            auto op = phaseop < Ti, To > (detx, dety, detz, k);
            for (int i = 0; i < Natoms; i++) {
                Ti4 atom = __ldg( & atoms[i]);
               
                auto phase = op(atom);
                /* kahan summation to give meaningful result for Natoms>1e4 in single precision */
                auto yx = phase.x - cx;
                auto yy = phase.y - cy;
                auto tx = accumx + yx;
                auto ty = accumy + yy;
                cx = (tx - accumx) - yx;
                cy = (ty - accumy) - yy;
                accumx = tx;
                accumy = ty;
            }
            ret[x * maxy + y] = Vec < To, 2 > ::make((To) accumx, (To) accumy);
        }
    };
};

extern "C"
__global__ void wfkernelf(float2 * __restrict__ ret, const float4 * __restrict__ atoms, double detz, double pixelsize, double k, int maxx, int maxy, int Natoms) {
    wfkernel < float, float > ()(ret, atoms, detz, pixelsize, k, maxx, maxy, Natoms);
};

extern "C"
__global__ void wfkerneld(double2 * __restrict__ ret, const double4 * __restrict__ atoms, double detz, double pixelsize, double k, int maxx, int maxy, int Natoms) {
    wfkernel < double, double > ()(ret, atoms, detz, pixelsize, k, maxx, maxy, Natoms);
};

"""

import numpy as _np
import cupy as _cp
from time import sleep as _sleep

def _pinned(shape, dtype):
    size = _np.prod(shape)
    mem = _cp.cuda.alloc_pinned_memory(size * _np.dtype(dtype).itemsize)
    ret = _np.frombuffer(mem, dtype, size).reshape(shape)
    return ret


def simulate_gen(simobject, Ndet, pixelsize, detz, k, settings="double", init=True, maximg=_np.inf, *args, **kwargs):
    """
    returns an array of simulated wavefields
    parameters:
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
    init: do full initialisation and asynch start of first calculation on generator creation
    maximg: generate StopIteration after maximg images.
    """

    if "single" in settings:
        intype, outtype, kernelname = _np.float32, _np.complex64, "wfkernelf"
    else:
        intype, outtype, kernelname = _np.float64, _np.complex128, "wfkerneld"
    options = []
    if not "scale" in settings:
        options += ["-Dnodist"]
    if "nf" in settings:
        options += ["-Dusenf"]
    if "secondorder" in settings:
        options += ["-Dsecondorder"]
    if not "nofast" in settings:
        options += ["--use_fast_math"]

    if _np.size(Ndet) == 1:
        Ndet = [Ndet, Ndet]
    maxx, maxy = Ndet
    threadsperblock = (16, 16, 1)
    blockspergrid_x = int(_np.ceil(Ndet[0] / threadsperblock[0]))
    blockspergrid_y = int(_np.ceil(Ndet[1] / threadsperblock[1]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    module = _cp.RawModule(code=code, backend="nvcc", options=tuple(["--std=c++11", "-O3", "--restrict"] + options))
    kernel = module.get_function(kernelname)
    
    d_wf = _cp.zeros((Ndet[0], Ndet[1]), dtype=outtype)
    d_atoms = _cp.zeros((simobject.N, 4), intype)
    h_atoms = _pinned((simobject.N, 4), intype)
    def _gen():
        h_atoms[:,:3], h_atoms[:,3:] = simobject.get2()
        d_atoms.set(h_atoms)    
        kernel(blockspergrid, threadsperblock, (d_wf, d_atoms, float(detz), float(pixelsize), float(k), int(Ndet[0]), int(Ndet[1]), int(simobject.N)))
        count = 1
        while True:
            if count == maximg: 
                yield d_wf.get()
            elif count > maximg:
                return
            else:
                h_atoms[:,:3], h_atoms[:,3:] = simobject.get2()
                d_atoms.set(h_atoms)    
                ret=d_wf.get()
                kernel(blockspergrid, threadsperblock, (d_wf, d_atoms, float(detz), float(pixelsize), float(k), int(Ndet[0]), int(Ndet[1]), int(simobject.N)))
                yield ret
            count += 1

    return _gen()


def simulate(Nimg, simobject, Ndet, pixelsize, detz, k, settings="double", verbose=True, *args, **kwargs):
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
    """
    if _np.size(Ndet) == 1:
        Ndet = [Ndet, Ndet]
    gen=simulate_gen(simobject, Ndet, pixelsize, detz, k, settings=settings, init=True, maximg=Nimg)
    result = _np.empty((Nimg, Ndet[0], Ndet[1]), _np.complex128)
    try:
        for i,img in enumerate(gen):
            if verbose: 
                print(i, end='. ', flush=True)
            result[i,...]=img
        return result
    except KeyboardInterrupt:
        if verbose:
            print('Interrupted')
        return result[:i]