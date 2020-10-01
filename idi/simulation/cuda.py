from __future__ import division as _future_div, print_function as _future_print
from six import print_ as _print
import numpy as _np
import pycuda.driver
import pycuda.autoinit
from pycuda.compiler import SourceModule as _SrcMod, DEFAULT_NVCC_FLAGS
from jinja2 import Template as _Template


def wavefield_kernel_double(Natoms, Ndet, pixelsize, detz, k):
    '''
    returns a cuda implementation of the wavefield, used internally
    Natoms: Number of atoms
    Ndet: detector pixels
    pixelsize: detector pixelsize
    detz: detector distance
    k: angular wavenumber
    returns a function with signature (ret float2[:], atompositionsandphases double4[:]) 
        that will write the wavefield into ret (real,imag)
        
    USES A LOT OF DOUBLE PRECISON
    '''
    maxx, maxy = Ndet
    tpl = _Template(
        """
    __global__ void wfkernel( double2* __restrict__ ret, const double4* __restrict__  atoms)
    {
        int x = blockIdx.x*blockDim.x + threadIdx.x;
        int y = blockIdx.y*blockDim.y + threadIdx.y;  
        int reti = x*{{ maxy }} + y;
        //const double TWOPI = 6.283185307179586476925287;
        if ((x < {{ maxx }}) && (y < {{ maxy }}))
        {
            double detx = (x-{{ maxx }}/2)*{{ pixelsize }};
            double dety = (y-{{ maxy }}/2)*{{ pixelsize }};
            double2 wf = double2(0,0);
            ret[reti] = double2(0,0);
            for (int i = 0; i < {{ Natoms }}; i++)
            {
                double4 atom = atoms[i]
                double dist = norm3d((
                    (detx-atom.x),
                    (dety-atom.y),
                    ({{ detz }}-atom.z)
                    );
                double rdist = 1/dist;
                double phase = dist*{{ k }}+atom.w);
                double real, imag;
                sincos(phase, &imag, &real);
                wf.x += real*rdist;
                wf.y += imag*rdist;
            }
            ret[reti]  = wf;
        }
    }
    """
    )
    src = tpl.render(maxx=maxx, maxy=maxy, pixelsize=pixelsize, Natoms=int(Natoms), detz=detz, k=k, shared_memory=shared_memory)
    # print(src)
    mod = _SrcMod(src)

    wfkernel = mod.get_function('wfkernel')
    return wfkernel

def wavefield_kernel_batches(Natoms, Ndet, pixelsize, detz, k):
    '''
    returns a cuda implementation of the wavefield, used internally
    Natoms: Number of atoms
    Ndet: detector pixels
    pixelsize: detector pixelsize
    detz: detector distance
    k: angular wavenumber
    returns a function with signature (ret float2[:], atompositionsandphases double4[:]) 
        that will write the wavefield into ret (real,imag)
    RUNS IN SINGLE BATCHES, ACCUMULATES IN DOUBLE
    '''
    maxx, maxy = Ndet
    tpl = _Template(
    """
    const double TWOPI = 6.283185307179586476925287;
    const double k = {{k}};
    const double detz = {{detz}};

    __device__ __forceinline__ float2 single(double4 atom, double detx, double dety)
    {
        double dist = norm3d(
                            ((double) detx - atom.x),
                            ((double) dety - atom.y),
                            (detz - atom.z)
                            );
        float rdist = 1 / __double2float_rn(dist);
        float phase = __double2float_rn(fmod(dist * k + atom.w, TWOPI));
        float real, imag;
        __sincosf(phase, &imag, &real);
        float2 wf = make_float2(real * rdist, imag * rdist);
        return wf;
    }

    __global__ void wfkernel(double2 * __restrict__ ret, const double4 * __restrict__ atoms)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        int reti = x*{{ maxy }} + y;
        if ((x < {{maxx}}) && (y < {{maxy}}))
        {
            double detx = (x - {{maxx}}/2) * {{pixelsize}};
            double dety = (y - {{maxy}}/2) * {{pixelsize}};
            double2 wf = make_double2(0., 0.);
            int i = 0;
            for (int i_batches = 0; i_batches < {{Natoms//2048}}; i_batches++)
            {
                float2 batch_wf = make_float2(0.f, 0.f);
                for (int i_inbatch = 0; i_inbatch < 2048; i_inbatch++)
                {
                    float2 cwf = single(atoms[i++], detx, dety);
                    batch_wf.x += cwf.x;
                    batch_wf.y += cwf.y;
                }
                wf.x += (double) batch_wf.x;
                wf.y += (double) batch_wf.y;
            }
            float2 batch_wf = make_float2(0.f, 0.f);
            while (i < {{Natoms}})
            {
                float2 cwf = single(atoms[i++], detx, dety);
                batch_wf.x += cwf.x;
                batch_wf.y += cwf.y;
            }
            wf.x += (double) batch_wf.x;
            wf.y += (double) batch_wf.y;
            ret[reti] = wf;
            }
    }
    """
    )
    src = tpl.render(maxx=maxx, maxy=maxy, pixelsize=pixelsize, Natoms=int(Natoms), detz=detz, k=k)
    #for i,line in enumerate(src.splitlines()):
    #    print(i,':\t',line)
        
    mod = _SrcMod(src,options=DEFAULT_NVCC_FLAGS + ['--use_fast_math'])
    wfkernel = mod.get_function('wfkernel')
    return wfkernel

def wavefield_kernel_old(Natoms, Ndet, pixelsize, detz, k):
    '''
    returns a cuda implementation of the wavefield, used internally
    Natoms: Number of atoms
    Ndet: detector pixels
    pixelsize: detector pixelsize
    detz: detector distance
    k: angular wavenumber
    returns a function with signature (ret float2[:], atompositionsandphases double4[:]) 
        that will write the wavefield into ret (real,imag)
    '''
    maxx, maxy = Ndet
    tpl = _Template(
        """
    __global__ void wfkernel( double2* __restrict__ ret, const double4* __restrict__  atom)
    {
        int x = blockIdx.x*blockDim.x + threadIdx.x;
        int y = blockIdx.y*blockDim.y + threadIdx.y;
        int reti = x*{{ maxy }} + y;
        if ((x < {{ maxx }}) && (y < {{ maxy }}))
        {
            const double PI = 3.141592653589793238463;
            int detx = (x-{{ maxx }}/2)*{{ pixelsize }};
            int dety = (y-{{ maxy }}/2)*{{ pixelsize }};
            float2 wf;
            wf.x = 0;
            wf.y = 0;
            for (int i = 0; i < {{ Natoms }}; i++)
            {
                double dist = norm3d((
                    (double)detx-atom[i].x),
                    ((double)dety-atom[i].y),
                    ({{ detz }}-atom[i].z)
                    );
                float rdist = 1/__double2float_rn(dist);
                //float phase = __double2float_rn((dist-(int)dist)*{{ k }}+atom[i].w);
                float phase = __double2float_rn(fmod(dist*{{ k }},2*PI)+atom[i].w);
                float real;
                float imag;
                __sincosf(phase, &imag, &real);
                wf.x += real*rdist;
                wf.y += imag*rdist;
            }
            ret[reti].x = (double) wf.x;
            ret[reti].y = (double) wf.y;
        }
    }
    """
    )
    src = tpl.render(maxx=maxx, maxy=maxy, pixelsize=pixelsize, Natoms=int(Natoms), detz=detz, k=k)
    # print(src)
    mod = _SrcMod(src)

    wfkernel = mod.get_function('wfkernel')
    return wfkernel

wavefield_kernel=wavefield_kernel_batches

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
    threadsperblock = (16, 16, 1)
    blockspergrid_x = int(_np.ceil(Ndet[0] / threadsperblock[0]))
    blockspergrid_y = int(_np.ceil(Ndet[1] / threadsperblock[1]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    h_wf1 = _np.empty((Ndet[0], Ndet[1], 2), dtype=_np.float64)
    d_wf1 = pycuda.driver.mem_alloc(h_wf1.nbytes)
    fwavefield = wavefield_kernel(simobject.N, Ndet, pixelsize, detz, k)
    d_atoms1 = pycuda.driver.mem_alloc(32 * simobject.N)

    for n in range(0, Nimg):
        if verbose:
            _print(n, end='', flush=True)
        h_atoms1 = simobject.get()
        pycuda.driver.memcpy_htod(d_atoms1, h_atoms1)
        if verbose:
            _print('.', end='', flush=True)
        fwavefield(d_wf1, d_atoms1, block=threadsperblock, grid=blockspergrid)
        pycuda.driver.memcpy_dtoh(h_wf1, d_wf1)
        result[n, ...] = h_wf1.view(dtype=_np.complex128)[..., 0]
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
    threadsperblock = (16, 16, 1)
    blockspergrid_x = int(_np.ceil(Ndet[0] / threadsperblock[0]))
    blockspergrid_y = int(_np.ceil(Ndet[1] / threadsperblock[1]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    h_wf1 = _np.empty((Ndet[0], Ndet[1], 2), dtype=_np.float32)
    d_wf1 = pycuda.driver.mem_alloc(h_wf1.nbytes)
    fwavefield = wavefield_kernel(simobject.N, Ndet, pixelsize, detz, k)
    d_atoms1 = pycuda.driver.mem_alloc(32 * simobject.N)
    h_atoms1 = simobject.get()

    while True:
        pycuda.driver.memcpy_htod(d_atoms1, h_atoms1)
        fwavefield(d_wf1, d_atoms1, block=threadsperblock, grid=blockspergrid)
        h_atoms1 = simobject.get()
        pycuda.driver.memcpy_dtoh(h_wf1, d_wf1)
        result = _np.copy(h_wf1.view(dtype=_np.complex64)[..., 0])
        yield result