import cupy as _cp
import numpy as _np
from pathlib import Path as _Path


code = (_Path(__file__).parent / 'simtime.cu').read_text()


def simulate(simobject, Ndet, pixelsize, detz, k, c, tau, pulsewidth, settings='mixed', threads=None):
    """
    Time dependent simulation with decaying amplitudes.
    simobject: simobject to use for simulation (in lengthunit)
    pixelsize: pixelsize (in lengthunit)
    detz: Detector-sample distance
    k: angular wave number (in 1/lengthunit)
    c: speed of light in (lengthunit/timeunit)
    tau: decay time (in timeunit)
    pulsewidth: FWHM of gaussian exciation pulse (in timeunit)
    settings: string, can contain
        double,single,mixed - precision
        nf - for nearfield form
        scale - do 1/r intensity scaling
    first call with new settings might recompile and take a few seconds
    """

    if 'double' in settings:
        nametmp, namesim = "tempsized", "simulated"
        ttype, atype, inouttype = _np.float64, _np.complex128, _np.float64
    elif 'single' in settings:
        nametmp, namesim = "tempsizef", "simulatef"
        ttype, atype, inouttype = _np.float32, _np.complex64, _np.float32
    else:  # mixed
        nametmp, namesim = "tempsizedf", "simulatedf"
        ttype, atype, inouttype = _np.float32, _np.complex128, _np.float64
    options = []
    if 'nf' in settings:
        options += ['-Dusenf']
    if 'scale' not in settings:
        options += ['-Dnodist']
    options += ['-dc', '--std=c++11', '--expt-relaxed-constexpr', '-O3', '--use_fast_math']

    module = _cp.RawModule(code=code, backend='nvcc', options=tuple(options))
    ftempsize = module.get_function(nametmp)
    fsimulate = module.get_function(namesim)
    N = simobject.N
    if threads is None:
        if N > 1e8:
            threads = 1
        elif N > 1e7:
            threads = 2
        elif N > 1e6:
            threads = 4
        else:
            threads = 8

    if _np.size(Ndet) == 1:
        Ndet = [Ndet, Ndet]

    det = _cp.array(
        _np.array(_np.meshgrid(pixelsize * (_np.arange(Ndet[0]) - (Ndet[0] / 2)), pixelsize * (_np.arange(Ndet[1]) - (Ndet[1] / 2)), detz)).T.reshape(
            -1, 3
        ),
        inouttype,
    )
    pdet = _cp.array([i.data.ptr for i in det], _np.uint64)

    tmpout = _cp.zeros(1, _np.int64)
    ftempsize(grid=(1,), block=(1,), args=(N, tmpout))
    tempsize = int(tmpout.get()[0])
    temp = [_cp.zeros(tempsize // 8 + 1, _np.int64) for i in range(threads)]
    ptemp = _cp.array([i.data.ptr for i in temp], _np.uint64)

    t = [_cp.zeros(N, ttype) for i in range(2 * threads)]
    a = [_cp.zeros(N, atype) for i in range(2 * threads)]
    pt = _cp.array([i.data.ptr for i in t], _np.uint64)
    pa = _cp.array([i.data.ptr for i in a], _np.uint64)
    data = _cp.array(simobject.get(), inouttype)
    times = _cp.array(_np.random.randn(simobject.N) * (pulsewidth / 2.35), inouttype)
    # _cp.random.seed(_np.random.randint(2**64-1,dtype=_np.uint64))
    # times = (_cp.random.randn(N,dtype=_np.float64)*(pulsewidth/2.35)).astype(inouttype) #cave: this uses different seed as numpy
    output = _cp.zeros(len(det), inouttype)
    poutput = _cp.array([i.data.ptr for i in output], _np.uint64)
    _cp.cuda.get_current_stream().synchronize()
    for start in range(0, len(pdet), threads):
        cthreads = int(min(len(pdet) - start, threads))
        end = start + cthreads
        fsimulate(
            grid=(1,),
            block=(cthreads,),
            args=(data, times, pdet[start:end], float(tau), float(c), float(k), int(N), int(cthreads), pt, pa, tempsize, ptemp, poutput[start:end],),
        )
    _cp.cuda.get_current_stream().synchronize()
    return output.get().reshape(Ndet)
