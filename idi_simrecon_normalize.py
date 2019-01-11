#!/bin/env python
import matplotlib.pyplot as plt
import numpy as np
from optparse import OptionParser
import idi.reconstruction as recon

parser = OptionParser(usage = "usage: %prog inputfile outputfile")
parser.add_option("-p", "--pmax", dest="pmax", type='int', default=8,
                      help="max. number of parallel processes (default 8)")

(options, args) = parser.parse_args()
if len(args) != 2:
    parser.error("incorrect number of arguments. Specify inputfile  and outputfile")
infile=args[0]
outfile=args[1]

data=np.load(infile)
settings=data['settings'].take(0)
Nimg=settings['Nimg']
z=(settings['detz']/100)/(settings['pixelsize']*1e-6*2**settings['bin'])


photonsum=data['photonsum']
sumcorr=recon.direct.corr(photonsum,z)
out=data['corr']/sumcorr*Nimg

np.savez_compressed(outfile,out=out,settings=settings)
print(settings)
