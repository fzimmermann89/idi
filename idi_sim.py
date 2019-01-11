#!/bin/env python
from __future__ import division, print_function
from six import print_ as print
import IPython
from optparse import OptionParser
import idi.simulation as sim
import numpy as np
from numpy import pi

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
parser.add_option('--nocuda', action="store_false", dest="cuda", default=True,
                  help='dont use cuda')   
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
    simobject=sim.simobj.sphere(Natoms,r,E)
    simobject.rndPhase=rndphase
    simobject.rndPos=rndpos
else:
    a=options.a*1e-4 #in um
    N=options.Nunitcells if options.Nunitcells is not None else Natoms
    if args[0]   == 'gridsc':    simobject=sim.simobj.gridsc(N,a,E,rotangles)
    elif args[0] == 'gridfcc':   simobject=sim.simobj.gridfcc(N,a,E,rotangles)
    elif args[0] == 'gridcuso4': simobject=sim.simobj.gridcuso4(N,a,E,rotangles)
    else: raise NotImplementedError("unknown object to simulate")
    simobject.rndPhase=rndphase
if options.cuda:
    result=sim.cuda.simulate(Nimg,simobject,Ndet,pixelsize,detz,k)
else:
    print("using cpu")
    result=sim.cpu.simulate(Nimg,simobject,Ndet,pixelsize,detz,k)
result=np.square(np.abs(result))
print('saving')
np.savez_compressed(outfile,result=result,settings=(vars(options),args))


