from __future__ import division as _future_div, print_function as _future_print
from six import print_ as _print
import numpy as _np
import pycuda.driver
import pycuda.autoinit
from pycuda.compiler import SourceModule as _SrcMod
from jinja2 import Template as _Template


def wavefield_kernel(Natoms, Ndet, pixelsize, detz, k):
    maxx = maxy = Ndet
    tpl = _Template(
    """
    __global__ void wfkernel( float2* __restrict__ ret, const double4* __restrict__  atom)
    {
        int x = blockIdx.x*blockDim.x + threadIdx.x;
        int y = blockIdx.y*blockDim.y + threadIdx.y;
        int reti = y*{{ maxx }} + x;
        if ((x < {{ maxx }}) && (y < {{ maxy }}))
        {
            const double PI =3.141592653589793238463;
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
            ret[reti].x = wf.x;
            ret[reti].y = wf.y;
        }
    }  
    """
    )
    src = tpl.render(maxx=Ndet, maxy=Ndet, pixelsize=pixelsize, Natoms=int(Natoms), detz=detz, k=k)
    # print(src)
    mod = _SrcMod(src)

    wfkernel = mod.get_function('wfkernel')
    return wfkernel


def simulate(Nimg, simobject, Ndet, pixelsize, detz, k, verbose=True):
    result = _np.empty((Nimg, Ndet, Ndet), dtype=complex)
    threadsperblock = (16, 16, 1)
    blockspergrid_x = int(_np.ceil(Ndet / threadsperblock[0]))
    blockspergrid_y = int(_np.ceil(Ndet / threadsperblock[1]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    h_wf1 = _np.empty((Ndet, Ndet, 2), dtype=_np.float32)
    d_wf1 = pycuda.driver.mem_alloc(h_wf1.nbytes)
    fwavefield = wavefield_kernel(simobject.N, Ndet, pixelsize, detz, k)
    d_atoms1 = pycuda.driver.mem_alloc(32 * simobject.N)

    for n in range(0, Nimg):
        if verbose: _print(n, end='', flush=True)
        h_atoms1 = simobject.get()
        pycuda.driver.memcpy_htod(d_atoms1, h_atoms1)
        if verbose: _print('.', end='', flush=True)
        fwavefield(d_wf1, d_atoms1, block=threadsperblock, grid=blockspergrid)
        pycuda.driver.memcpy_dtoh(h_wf1, d_wf1)
        result[n, ...] = h_wf1.view(dtype=_np.complex64)[..., 0]
        if verbose: _print('. ', end='', flush=True)
    return result
