#!/bin/env python
from __future__ import division, print_function
from six import print_ as print
import IPython
import numpy as np
import math
from numpy import sin,cos
import numexpr as ne
from numpy import pi
from optparse import OptionParser
import pycuda.driver as drv
import pycuda.autoinit
from pycuda.compiler import SourceModule, DEFAULT_NVCC_FLAGS
from jinja2 import Template
import numba

def wavefield_kernel(Natoms, Ndet, pixelsize, detz,k):
    maxx = maxy = Ndet
    tpl = Template("""
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
                //sincospif(phase, &imag, &real);
                __sincosf(phase, &imag, &real);                
                wf.x += real*rdist;
                wf.y += imag*rdist;
            }
            ret[reti].x =  wf.x;
            ret[reti].y = wf.y;
        }
    }  
    """)
    src = tpl.render(maxx=Ndet, maxy=Ndet,pixelsize=pixelsize,Natoms=int(Natoms),detz=detz,k=k)
    #print(src)
    mod = SourceModule(src,options=DEFAULT_NVCC_FLAGS+['-lineinfo'],keep=True)

    wfkernel = mod.get_function("wfkernel")
    return wfkernel


class atoms:
    def __init__(self, N, E, pos):
        self._N = int(N)
        self._E = np.ones(int(N))*E
        self._pos = pos
        self.rndPhase=False
    def get(self):
        raise NotImplementedError("abstract method")

    @property
    def N(self):
        return self._N

    @property
    def E(self):
        return self._E
    
    def get(self, rndPhase=True):
        k = 2*pi/(1.24/self._E)  # in 1/um
        z = self._pos[2,...]
        rnd = np.random.rand(self._N) if rndPhase else 0
        phase = ne.evaluate('(k*z+rnd*2*pi)%(2*pi)')
        ret = np.concatenate((self._pos, phase[np.newaxis,:]))        
        return ret


class sphere(atoms):
    def __init__(self, N, r, E):
        pos = self._rndSphere(r, N)
        atoms.__init__(self, N, E, pos)
        self._r=r
        self.rndPos=False
        
    @staticmethod
    def _rndSphere(r, N):
        rnd = np.random.rand(N)
        t = ne.evaluate('arcsin(2.*rnd-1.)')
        rnd = np.random.rand(N)
        p = ne.evaluate('2.*pi*rnd')
        rnd = np.random.rand(N)
        r = ne.evaluate('r*rnd**(1./3)')
        x = ne.evaluate('r * cos(t) * cos(p)')
        y = ne.evaluate('r * cos(t) * sin(p)')
        z = ne.evaluate('r * sin(t)')
        return np.array((x, y, z))

    def get(self, rndPhase=True, rndPos=False):
        k = 2*pi/(1.24/self._E)  # in 1/um
        if rndPos:
            self._pos = self._rndSphere(self._r, self._N)      
        return atoms.get(self,rndPhase)

class grid(atoms):
    def __init__(self, lconst,langle,unitcell,Ns,rotangles,E):
        pos=grid._lattice(lconst,langle,unitcell,Ns)
        if np.any(rotangles):
            self._rotmatrix = grid._rotation(*rotangles)
            pos = np.matmul(self._rotmatrix,pos) 
        else:
            self._rotmatrix = None    
        atoms.__init__(self, np.product(Ns)*len(unitcell), E, pos)

    @staticmethod
    def _lattice(lconst,langle,unitcell,repeats): 
        cosa,cosb,cosc = cos(np.array(langle)*pi/180.0)
        sina,sinb,sinc = sin(np.array(langle)*pi/180.0)
        basis = (np.array([[1,0,0],
                          [cosc, sinc,0],
                          [cosb, (cosa-cosb*cosc)/sinc, np.sqrt(sinb**2 - ((cosa-cosb*cosc)/sinc)**2)]])
                *np.expand_dims(lconst,1)
                )
        atoms = unitcell

        tmpatoms=[]
        for i in range(repeats[0]):
            offset = basis[0] * i
            tmpatoms.append(atoms + offset[np.newaxis,:])
        atoms = np.concatenate(tmpatoms)
        tmpatoms=[]
        for j in range(repeats[1]):
            offset = basis[1] * j
            tmpatoms.append(atoms + offset[np.newaxis,:])
        atoms = np.concatenate(tmpatoms)
        tmpatoms=[]
        for k in range(repeats[2]):
            offset = basis[2] * k
            tmpatoms.append(atoms + offset[np.newaxis,:])
        atoms = np.concatenate(tmpatoms)
        return (atoms-np.max(atoms,axis=0)/2.0).T


    @staticmethod
    @numba.njit
    def _rotation(alpha, beta, gamma):
#         #euler angles        
#         M = np.array([[cos(alpha)*cos(gamma)-sin(alpha)*cos(beta)*sin(gamma),
#                        sin(alpha)*cos(gamma)+cos(alpha)*cos(beta)*sin(gamma),
#                        sin(beta)*sin(gamma)],
#                       [-cos(alpha)*sin(gamma)-sin(alpha)*cos(beta)*cos(gamma),
#                        -sin(alpha)*sin(gamma)+cos(alpha)*cos(beta)*cos(gamma),
#                        sin(beta)*cos(gamma)],
#                       [sin(alpha)*sin(beta),
#                        -cos(alpha)*sin(beta),
#                        cos(beta)]])
        #yaw pitch roll
        M = np.array([
            [cos(beta)*cos(gamma),sin(alpha)*sin(beta)*cos(gamma)-cos(alpha)*sin(gamma),cos(alpha)*sin(beta)*cos(gamma)+sin(alpha)*sin(gamma)], 
            [cos(beta)*sin(gamma),sin(alpha)*sin(beta)*sin(gamma)+cos(alpha)*cos(gamma),cos(alpha)*sin(beta)*sin(gamma)-sin(alpha)*cos(gamma)],  
            [-sin(beta),sin(alpha)*cos(beta),cos(alpha)*cos(beta)]
        ])
        return M

    def get(self, rndPhase=True, rndOrientation=False ):
        if rndOrientation:
            m=grid._rotation(*2*pi*np.random.rand(3))
            self._pos=np.matmul(m,self._pos)
        return atoms.get(self,rndPhase)

    @property
    def n(self):
        return self._n


class gridsc(grid):
    def __init__(self, N, a, E,rotangles):
        if (np.array(N)).size==1:
            N = int(np.rint(N**(1/3.)))
            N=[N,N,N]
        lconst=[a,a,a]
        unitcell=[[0,0,0]]
        langle=[90,90,90]
        grid.__init__(self,lconst,langle,unitcell,N,rotangles,E)
            
class gridfcc(grid):
    def __init__(self, N, a, E,rotangles):
        if (np.array(N)).size==1:
            N = int(np.rint((N/4)**(1/3.)))
            N=[N,N,N]
        lconst=[a,a,a]
        unitcell=[[0,0,0],[.5,.5,0],[.5,.5,0],[.5,0,.5],[0,.5,.5]]
        langle=[90,90,90]
        grid.__init__(self,lconst,langle,unitcell,N,rotangles,E)
            
class gridcuso4(grid):
    def __init__(self, N, E,rotangles):
        N = int(np.rint((N/2)**(1/3.)))
        unitcell=[[0,0,0.5],[0,0.5,0]]
        lconst=np.array([0.60,0.61,1.07])*1e-4
        langle=np.array([77.3,82.3,72.6])
        Ns=[N,N,N]
        grid.__init__(self,lconst,langle,unitcell,Ns,rotangles,E)
            
def simulate(Nimg,simobject,Ndet,pixelsize,detz,k,verbose=True):
    result=np.empty((Nimg,Ndet,Ndet),dtype=complex)
    threadsperblock = (16, 16, 1)
    blockspergrid_x = int(math.ceil(Ndet / threadsperblock[0]))
    blockspergrid_y = int(math.ceil(Ndet / threadsperblock[1]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    h_wf1=np.empty((Ndet,Ndet,2),dtype=np.float32)
    d_wf1=drv.mem_alloc(h_wf1.nbytes)
    fwavefield = wavefield_kernel(simobject.N, Ndet, pixelsize, detz,k)
    d_atoms1 = drv.mem_alloc(32*simobject.N)

    for n in range(0, Nimg):
        if verbose: print(n, end='',flush=True)
        h_atoms1 = simobject.get()
        drv.memcpy_htod(d_atoms1, h_atoms1)
        if verbose: print('.', end='',flush=True)
        fwavefield(d_wf1,d_atoms1, block = threadsperblock, grid=blockspergrid)
        drv.memcpy_dtoh(h_wf1, d_wf1)
        result[n,...]=h_wf1.view(dtype=np.complex64)[...,0]
        if verbose: print('. ', end='',flush=True)
           
if __name__ == "__main__":      
    ##input parsing
    parser = OptionParser(usage = "usage: %prog [options] sphere/gridsc/gridfcc/gridblocks")
    parser.add_option("-d", "--Ndet", dest="Ndet",type='int', default=512,
                      help="Number of pixels on detector (default: 512)")
    parser.add_option("-r", dest="r", type='float',default=50,
                      help="radius in nm (default: 50)")
    parser.add_option("-a", dest="a", type='float',default=50,
                      help="lattice constant in A (default: 5)")
    parser.add_option("-n", "--Natoms",type='float', dest="Natoms", default=None,
                      help="Number of atoms (default: 1e5)")
    parser.add_option("-p", "--pixelsize", type='float',dest="pixelsize", default=75,
                      help="Size of pixels on detector in um (default: 75)")
    parser.add_option("-z", "--detz", dest="detz",type='float', default=10,
                      help="z distance of detector in cm(default: 10)")
    parser.add_option("-i", "--Nimg", dest="Nimg", type='int', default=100,
                      help="Number of images (default: 100)")
    parser.add_option("-e", "-E", dest="E", type='float', default=1500,
                      help="Energy in eV (default: 1500)")
    parser.add_option('-c','--coherent', action="store_false", dest="rndphase", default=True,
                      help='no random phases')
    parser.add_option("-o", "--output", dest="outfile", type='string', default="out.npz",
                      help="output file (default: out.npz)")
    def cb_Nunitcell(option, opt, value, parser):
        setattr(parser.values, option.dest, [int(x) for x in value.split(',')])
    parser.add_option("--Nunitcells", type='string', action='callback', dest="Nunitcells", default=None, callback=cb_Nunitcell,
                      help="grid: nx,ny,nz Number of unitcells in x y and z direction, overrides Natoms (default: use Natoms)")
    parser.add_option("--anglex",type='float', dest="ax", default=0,
                      help="grid: Rotation angle in degree (default: 0)")
    parser.add_option("--angley",type='float', dest="ay", default=0,
                      help="grid: Rotation angle in degree (default: 0)")
    parser.add_option("--anglez",type='float', dest="az", default=0,
                      help="grid: Rotation angle in degree (default: 0)")
    parser.add_option('-f','--fixedpositions', action="store_false", dest="rndpos", default=True,
                      help='sphere: no randomly changing positions')      
    (options, args) = parser.parse_args()
    if len(args) != 1:
        parser.error("incorrect number of arguments. Specify either sphere, gridsc, gridfcc or grudcuso4")
    if options.Nunitcells is not None and (options.Natoms is not None or args[0]=='sphere'):
        parser.error("Nunitcells is only allowed for grid* and if Natoms is not specified")
    
    outfile=options.outfile
    Natoms=int(options.Natoms) if options.Natoms is not None else int(1e5) 
    Ndet=int(options.Ndet)
    detz=options.detz*1e4 #in um
    pixelsize=options.pixelsize #in um
    Nimg=int(options.Nimg)
    E=options.E #in ev
    rndphase=options.rndphase
    rndpos=options.rndpos
    rotangles=np.array([options.ax,options.ay,options.az])/180*pi
    k = 2*pi/(1.24/E)  # in 1/um
    
    if args[0]=='sphere':
        r=options.r*1e-3 #in um
        simobject=sphere(Natoms,r,E)
        simobject.rndPhase=rndphase
        simobject.rndPos=rndpos
    else:
        a=options.a*1e-4 #in um
        N=options.Nunitcells if options.Nunitcells is not None else Natoms
        if args[0]   == 'gridsc':    simobject=gridsc(N,a,E,rotangles)
        elif args[0] == 'gridfcc':   simobject=gridfcc(N,a,E,rotangles)
        elif args[0] == 'gridcuso4': simobject=gridcuso4(N,a,E,rotangles)
        else: raise NotImplementedError("unknown object to simulate")
        simobject.rndPhase=rndphase
    result=simulate(Nimg,simobject,Ndet,pixelsize,detz,k)
    result=np.square(np.abs(result))
    np.savez_compressed(outfile,result=result,settings=(vars(options),args))


