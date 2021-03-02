import numba as _numba
from numba import cuda
import numpy as _np
import math
import contextlib
import warnings


@contextlib.contextmanager
def corrfunction(shape, z, maxq, xcenter=None, ycenter=None):
    """
    GPU based Autocorrelation with q correction

    parameters:
    shape (tuple) of inputs in pixels
    z (scalar) distance of detector in pixels
    maxq (scalar): maxmum distance
    optional:
    xcenter (scalar): position of center in x direction, defaults to shape[0]/2
    ycenter (scalar): position of center in x direction, defaults to shape[1]/2
    returns a function with signature float[:,:,:](float[:,:] image) that does the correlation
    and returns val(dqz,dqy,dqx)=sum_qx,qy image(qx,qy)*image(qx+dqx,qy+dqy)
    """

    stream = None
    dzd = None
    dvals = None
    doutput = None
    jkernel = None

    try:
        cuda.select_device(0)
        stream = cuda.stream()
        Nr, Nc = int(shape[0]), int(shape[1])
        if not 32 <= maxq <= 1024:
            warnings.warn("maxq will be clamped between 32 and 1024")
        qmax = _numba.int32(max(32, min(1024, maxq, Nr, Nc)))
        y, x = _np.meshgrid(_np.arange(Nc, dtype=_np.float64), _np.arange(Nr, dtype=_np.float64))
        xcenter = _numba.float32(xcenter or Nr / 2)
        ycenter = _numba.float32(ycenter or Nc / 2)

        x -= xcenter
        y -= ycenter
        d = _np.sqrt(x ** 2 + y ** 2 + z ** 2)
        zd = _np.array((z / d), _np.float32, order="C")
        maxdqz = _numba.int32(math.ceil(z * (_np.max(zd) - _np.min(zd)))) | 1
        cz = _numba.float32(z)

        with stream.auto_synchronize():
            dzd = _numba.cuda.to_device(zd, stream)
            doutput = _numba.cuda.device_array((2 * maxdqz, 2 * qmax, 2 * qmax), _np.float32, stream=stream)

        del zd, x, y, d

        def kernel(zd, val, out):
            refr = _numba.int32(_numba.cuda.blockIdx.y)
            offsetr = _numba.int32(_numba.cuda.blockIdx.z)
            # offsetc=numba.cuda.threadIdx.x-qmax
            refzd = _numba.cuda.shared.array(Nc, _numba.float32)
            refv = _numba.cuda.shared.array(Nc, _numba.float32)
            tzd = _numba.cuda.shared.array(Nc, _numba.float32)
            tv = _numba.cuda.shared.array(Nc, _numba.float32)
            tr = _numba.int32(refr + offsetr)
            if not (0 <= tr < Nr and 0 <= refr < Nr):
                return
            loadid = _numba.int32(_numba.cuda.threadIdx.x)
            _numba.cuda.syncthreads()
            while loadid < Nc:
                refzd[loadid] = zd[refr, loadid]
                refv[loadid] = val[refr, loadid]
                tzd[loadid] = zd[tr, loadid]
                tv[loadid] = val[tr, loadid]
                loadid += _numba.cuda.blockDim.x
            _numba.cuda.syncthreads()
            tr2 = _numba.float32(tr) - xcenter
            refr2 = _numba.float32(refr) - xcenter
            offsetc = -_numba.int32(_numba.cuda.threadIdx.x)
            while offsetc < qmax:
                for refc in range(0, Nc):
                    tc = _numba.int32(refc + offsetc)
                    if 0 <= tc < Nc:
                        dqx = qmax + _numba.int32(round(tr2 * tzd[tc] - refr2 * refzd[refc]))
                        dqy = qmax + _numba.int32(round((_numba.float32(tc) - ycenter) * tzd[tc] - (_numba.float32(refc) - ycenter) * refzd[refc]))
                        dqz = maxdqz + _numba.int32(round(cz * (refzd[refc] - tzd[tc])))
                        _numba.cuda.atomic.add(out, (dqz, dqy, dqx), refv[refc] * tv[tc])
                offsetc += _numba.int32(_numba.cuda.blockDim.x)

        def assemble(vals):
            id0, id1, id2 = _numba.cuda.grid(3)
            vals[id0, id1, id2] += vals[-id0, -id1, -id2]
            vals[-id0, -id1, -id2] = vals[id0, id1, id2]

        jkernel = _numba.cuda.jit("void(float32[:,:],float32[:,:],float32[:,:,:])", fastmath=True)(kernel)
        jkernel = jkernel[(1, Nr, qmax), (qmax, 1, 1), stream]
        jassemble = _numba.cuda.jit("void(float32[:,:,:])", fastmath=True)(assemble)
        jassemble = jassemble[(doutput.shape[0], doutput.shape[1], 1), (1, 1, doutput.shape[2] // 2), stream]

        def corr(input):
            """
            Do the correlation
            """
            if stream is None:
                raise ValueError("already closed, use within with statement")
            with stream.auto_synchronize():
                dvals = cuda.to_device(input.astype(_np.float32), stream)
                doutput[:, :, :] = 0
                jkernel(dzd, dvals, doutput)
                jassemble(doutput)
                return doutput.copy_to_host(stream=stream).astype(_np.float64)

        yield corr
    finally:
        if stream is not None:
            stream.synchronize()
        stream = dzd = dvals = doutput = jkernel = jassemble = None
        for x in locals():
            del x
        import gc

        gc.collect()
