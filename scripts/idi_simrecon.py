#!/bin/env python
import numpy as np
from optparse import OptionParser
import numba
from idi.reconstruction.ft import corr
from idi.simulation.common import randomphotons
from idi.util import *

pmax = 8  # todo
numba.config.NUMBA_NUM_THREADS = pmax

parser = OptionParser(usage="usage: %prog [options] inputfile")
parser.add_option('-N', '--Nphotons', dest='Nphotons', type='float', default=1e3,
                  help="Number of Photons (default 1e3)")
parser.add_option('-o', '--output', dest='outfile', type='string', default='',
                  help="output file (default: (inputfile)-(Nphotons)-out.npz)")
parser.add_option('-p', '--pmax', dest='pmax', type='int', default=8,
                  help="max. number of parallel processes (default 8)")
parser.add_option('-b', '--bin', dest='bin', type='int', default=0,
                  help="bin 4^n pixels (default 0)")

(options, args) = parser.parse_args()
if len(args) != 1:
    parser.error("incorrect number of arguments. Specify inputfile")
infile = args[0]
Nphotons = int(options.Nphotons)
pmax = options.pmax
nrebin = options.bin
outfile = (
    '%s-%i-out.npz' % (infile if nrebin == 0 else infile + '-b%i' % nrebin, Nphotons)
    if options.outfile == ''
    else options.outfile
)

incoherent = np.load(infile)['result']
incoherent = np.square(np.abs(incoherent))
simsettings = np.load(infile)['settings']
simtype = simsettings[1]
simsettings = simsettings[0]
pixelsize = simsettings['pixelsize'] * 1e-6  # in m
pixelsize = pixelsize * 2 ** nrebin
dz = simsettings['detz'] / 100  # in m
z = dz / pixelsize

photons = np.asarray(randomphotons(rebin(incoherent, nrebin), Nphotons)).astype(np.uint32)
out = corr(photons, z)
photonsum = np.sum(photons, axis=0)
np.savez_compressed(
    outfile,
    corr=out,
    photonsum=photonsum,
    simsettings=vars(simsettings),
    simtype=simtype,
    reconsettings=(vars(options), args),
)
