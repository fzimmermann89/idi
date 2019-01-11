#!/bin/env python
import matplotlib.pyplot as plt
import numpy as np
from optparse import OptionParser
from reconrad import *

parser = OptionParser(usage = "usage: %prog inputfile outputfile")
parser.add_option("-p", "--pmax", dest="pmax", type='int', default=12,
                      help="max. number of parallel processes (default 12)")

(options, args) = parser.parse_args()
if len(args) != 2:
    parser.error("incorrect number of arguments. Specify inputfile  and outputfile")
pmax=options.pmax
infile=args[0]
outfile=args[1]
print(pmax)
data=np.load(infile)
settings=data['settings'].take(0)
Nimg=settings['Nimg']
z=(settings['detz']/100)/(settings['pixelsize']*1e-6*2**settings['bin'])


photonsum=data['photonsum']
print('correlating')
sumcorr=radcorr(photonsum,z)
print('normalizing')
out=data['corr']/sumcorr*Nimg
print('saving')
np.savez_compressed(outfile,out=out,settings=settings)
print(settings)
