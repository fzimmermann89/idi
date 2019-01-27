from __future__ import division, print_function
from six import print_ as print
import psana as ps
import numpy as np
import idi.util
import idi.reconstruction as recon
import scipy.ndimage as snd
import scipy.signal as ss
import signal

def getmask(detector,run):
    #get mask for detector and run with masked neighbours
    mask = detector.image(run,nda_in=detector.mask_geo(run))
    mask = detector.mask_neighbors(mask,True)
    return mask.astype(bool)

def filter(inp,mask):
    #fill masked
    ret=inp.copy()
    ret[~mask.astype(bool)]=np.nan
    ret=idi.util.fill(ret)
    #medianfilter
    ret=ss.medfilt2d(ret,3)
    return ret

def offsetpad(inp,offsets):
    pad=[(0,0)]*(inp.ndim-len(offsets)) + [(-2*offset if offset<0 else 0, 2*offset if offset>0 else 0) for offset in offsets]
    return np.pad(inp,pad,mode='constant')


#todo
from optparse import OptionParser
parser = OptionParser()
(options, args) = parser.parse_args()
run=int(args[0])
pixelsize=75*1e-6
detz=4.1*1e-2
E=8.04
thres=2e6
nimages_offset=100
nimages_recon=np.inf
z=detz/pixelsize
outfile='/reg/d/psdm/xpp/xppx37817/scratch/zimmf/%s.npz'%run


ds = ps.DataSource('exp=xppx37817:run=%s:smd'%run)
jungfrau=ps.Detector('jungfrau1M')
mask=getmask(jungfrau,run)  

#get images for center detection
print('getting images for offset')
imagesum=0
nimages=0
for n,evt in enumerate(ds.events()):
        image=jungfrau.image(evt)
        if image is not None:
            s=np.sum(image)
            if s>thres:
                imagesum=imagesum+idi.util.photons(image,E)
                nimages+=1
        if nimages>=nimages_offset: break
print('finding offset')         
offset = idi.util.find_center(filter(imagesum,mask),mask)
print("Offset: %s"%str(offset))

#pad mask
mask=offsetpad(mask,offset)

#really lazy way to figure out size of output
res=np.zeros_like(recon.ft.corr(offsetpad(image,offset),z))

#setup save handler
import IPython
def handler(signum, frame):
    def dummy():
        print("please wait..")
        
    print("Calculation Stopped!")
    print("Received signal %i" % signum)
    signal.signal(signal.SIGTERM, dummy)
    signal.signal(signal.SIGINT, dummy)
    import sys
    try:
        print("Plz let me first save current progress..")
        np.savez(outfile, res=res, imagesum=imagesum, nimages=nimages)          
        print("saved! exiting now.")
    except:
        print("error writing output!")
        IPython.embed()
    sys.exit()
signal.signal(signal.SIGTERM, handler)
signal.signal(signal.SIGINT, handler)



#get images for correlation
print()
print('starting reconstruction')  
imagesum=0
nimages=0
ds = ps.DataSource('exp=xppx37817:run=%s:smd'%run)
for n,evt in enumerate(ds.events()):
        image=jungfrau.image(evt)
        if image is not None:
            s=np.sum(image)
            if s>thres:
                #got image, get photons
                nimages+=1
                image=offsetpad(image, offset)
                image[~mask]=0
                image=idi.util.photons(image, E)
                imagesum=imagesum+image
                #do correlation
                corr=recon.ft.corr(image, z)
                #add inplace
                np.add(res, corr, out=res)
                if nimages%10 == 0: 
                     print(nimages, end=' ', flush=True)
            if nimages>=nimages_recon: break  

#saving
np.savez(outfile, res=res, imagesum=imagesum, nimages=nimages)          



