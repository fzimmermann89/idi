from __future__ import division, print_function

# set the threading layer before any parallel target compilation
# import numba
# numba.config.THREADING_LAYER = 'tbb'

from six import print_ as print
import numpy as np
import psana as ps
import idi.util
import idi.reconstruction as recon
import scipy.ndimage as snd
import scipy.signal as ss
import signal
import time
import random

# import IPython
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)  # supress h5py warning
import gc


def getmask(detector, run):
    # get mask for detector and run with masked neighbours
    mask = detector.image(run, nda_in=detector.mask_geo(run))
    # mask = detector.mask_neighbors(mask,True)
    el = np.zeros([3, 3])
    el[..., 1] = 1
    mask = ~snd.morphology.binary_dilation(~(mask.astype(bool)), el, iterations=2)
    return mask.astype(bool)


def filter(inp, mask):
    # fill masked
    ret = inp.copy()
    ret[~mask.astype(bool)] = np.nan
    ret = idi.util.fill(ret)
    # medianfilter
    ret = ss.medfilt2d(ret, 3)
    return ret


def offsetpad(inp, offsets):
    pad = [(0, 0)] * (inp.ndim - len(offsets)) + [
        (-2 * offset if offset < 0 else 0, 2 * offset if offset > 0 else 0) for offset in offsets
    ]
    return np.pad(inp, pad, mode='constant')


# todo: parse command line options
from optparse import OptionParser

parser = OptionParser("usage: %prog [options] exp run")
parser.add_option('-z', dest='detz', default=4.1, type='float', help="detz in cm. default 4.1")
parser.add_option('-n', dest='nimages', default=0, type='int', help="limit number of images")
parser.add_option('-E', dest='E', default=8.04, type='float', help="energy in kev. default 8.04")
parser.add_option(
    '-t', dest='thres', default=1e6, type='float', help="sum filter threshold. default 1e6"
)
parser.add_option(
    '-p', dest='pixelsize', default=75, type='float', help="pixelsize in um. default 75"
)
parser.add_option('-o', dest='outpath', default=".", type='string', help="outpath")
(options, args) = parser.parse_args()
if len(args) != 2:
    parser.print_usage()
    exit()
exp = str(args[0])
run = int(args[1])
pixelsize = options.pixelsize * 1e-6  # m
detz = options.detz * 1e-2  # m
thres = options.thres  # sum keV
nimages_max = np.inf if options.nimages == 0 else options.nimages
z = detz / pixelsize
E = options.E  # kev
outfile = '%s/%s-%i.npz' % (options.outpath, exp, run)
print("%s Run %i. Output: %s" % (exp, run, outfile))
print(options, args)
# sleep random time to desync jobs
sleep = random.randint(0, 10)
print("sleeping for %i seconds" % sleep)
time.sleep(sleep)

ds = ps.DataSource('exp=%s:run=%i:smd' % (exp, run))
jungfrau = ps.Detector('jungfrau1M')
mask = getmask(jungfrau, run)

# get mean of images for normalisation and center detection
print('getting images for normalisation and offset')
imagesum = 0
nimages_sum = 0
for n, evt in enumerate(ds.events()):
    image = jungfrau.image(evt)
    if image is not None:
        s = np.sum(image)
        if s > thres:
            image[~mask] = 0
            imagesum = imagesum + idi.util.photons(image, E)
            nimages_sum += 1
            if nimages_sum % 10 == 0:
                print("%i/%i" % (nimages_sum, n + 1), end=' ', flush=True)
    if nimages_sum >= nimages_max:
        break
print()
print('finding offset..')
offset = idi.util.find_center(filter(imagesum, mask), mask)
print("Offset: %s" % str(offset))

# mask out bad pixels
mask[imagesum > (np.mean(imagesum) + 6 * np.std(imagesum))] = False
mask[imagesum <= 0] = False


# pad
mask = offsetpad(mask, offset)
imagesum = offsetpad(imagesum, offset)


# really lazy way to figure out size of output
res = np.zeros_like(recon.ft.corr(offsetpad(image, offset), z))
print("allocated result array (%ix%ix%i)" % res.shape)


# setup save handler
def handler(signum, frame):
    def dummy(signum, frame):
        print("please wait..")

    print("Calculation Stopped!")
    print("Received signal %i" % signum)
    signal.signal(signal.SIGTERM, dummy)
    signal.signal(signal.SIGINT, dummy)
    import sys

    try:
        print("Plz let me first save current progress..")
        settings = {
            'exp': exp,
            'run': run,
            'pixelsize': 75 * 1e-6,
            'detz': detz,
            'thres': thres,
            'nimages_max': nimages_max,
            'E': E,
        }
        np.savez(
            outfile,
            res=res,
            imagesum=imagesum,
            nimages=nimages,
            offset=offset,
            mask=mask,
            setting=settings,
        )
        print("saved! exiting now.")
    except:
        print("error writing output!")
        # IPython.embed()
    sys.exit()


signal.signal(signal.SIGTERM, handler)
signal.signal(signal.SIGINT, handler)
print("kill handler set")


# get images for correlation
print('starting reconstruction')
nimages = 0
ds = ps.DataSource('exp=xppx37817:run=%s:smd' % run)
for n, evt in enumerate(ds.events()):
    image = jungfrau.image(evt)
    if image is not None:
        s = np.sum(image)
        if s > thres:
            # got image, normalize and get photons
            nimages += 1
            image = offsetpad(image, offset)
            image[~mask] = 0
            image = idi.util.photons(image, E)
            np.divide(image, imagesum / nimages_sum, out=image, where=mask)
            # do correlation
            corr = recon.ft.corr(image, z)
            # add inplace
            np.add(res, corr, out=res)
            # cleanup
            corr = 0
            gc.collect()
            if nimages % 10 == 0:
                print("%i/%i" % (nimages, n + 1), end=' ', flush=True)
        if nimages >= nimages_max:
            break


# saving
print('saving')
settings = {
    'exp': exp,
    'run': run,
    'pixelsize': pixelsize,
    'detz': detz,
    'thres': thres,
    'nimages_max': nimages_max,
    'E': E,
}
np.savez(
    outfile, res=res, imagesum=imagesum, nimages=nimages, offset=offset, mask=mask, setting=settings
)
print('done')
