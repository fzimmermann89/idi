import numba as _numba
from numba import cuda
import numpy as _np
import math
import contextlib
import warnings


@contextlib.contextmanager
def corrfunction(shape, z, maxq, xcenter=None, ycenter=None):
    """
    GPU based radial Autocorrelation with q correction

    parameters:
    shape (tuple) of inputs in pixels
    z (scalar) distance of detector in pixels
    maxq (scalar): maxmum distance
    optional:
    xcenter (scalar): position of center in x direction, defaults to shape[0]/2
    ycenter (scalar): position of center in x direction, defaults to shape[1]/2
    returns a function with signature float[:](float[:,:] image) that does the correlation
    """
    stream = None
    saveblocks = _numba.int32(16)
    try:
        if not 15 <= maxq <= 511:
            warnings.warn("maxq will be clamped between 15 and 511")
        cuda.select_device(0)
        stream = cuda.stream()
        qmax = _numba.int32(max(15, min(511, maxq)))
        Nr, Nc = int(shape[0]), int(shape[1])
        y, x = _np.meshgrid(_np.arange(Nc, dtype=_np.float64), _np.arange(Nr, dtype=_np.float64))
        xcenter = _numba.float32(xcenter or Nr / 2)
        ycenter = _numba.float32(ycenter or Nc / 2)
        x -= xcenter
        y -= ycenter
        d = _np.sqrt(x ** 2 + y ** 2 + z ** 2)
        qx, qy, qz = ((k * z / d).astype(_np.float32) for k in (x, y, z))
        with stream.auto_synchronize():
            dqx = _numba.cuda.to_device(qx, stream)
            dqy = _numba.cuda.to_device(qy, stream)
            dqz = _numba.cuda.to_device(qz, stream)
            doutput = _numba.cuda.device_array((saveblocks, qmax + 1), _np.float64, stream=stream)

        del x, y, d, qx, qy, qz
        # shared memory size is limited, don't cache if rows are to large
        # instead, we could load them blockwise..
        shareval = shape[1] <= 4096
        sharerefq = shape[1] <= 2048
        sharetq = shape[1] <= 1024

        def kernel(qx, qy, qz, val, out):
            refr = _numba.cuda.blockIdx.y
            offsetr = _numba.cuda.blockIdx.z

            saveblock = _numba.int32(_numba.cuda.threadIdx.x % saveblocks)
            tr = _numba.int32(refr + offsetr)

            if not (0 <= tr < Nr and 0 <= refr < Nr):
                return

            if sharerefq:
                refq0, refq1, refq2 = (
                    _numba.cuda.shared.array(Nc, _np.float32),
                    _numba.cuda.shared.array(Nc, _np.float32),
                    _numba.cuda.shared.array(Nc, _np.float32),
                )
            else:
                refq0, refq1, refq2 = qx[refr, :], qy[refr, :], qz[refr, :]
            if sharetq:
                tq0, tq1, tq2 = (
                    _numba.cuda.shared.array(Nc, _np.float32),
                    _numba.cuda.shared.array(Nc, _np.float32),
                    _numba.cuda.shared.array(Nc, _np.float32),
                )
            else:
                tq0, tq1, tq2 = qx[tr, :], qy[tr, :], qz[tr, :]
            if shareval:
                refv = _numba.cuda.shared.array(Nc, _np.float32)
                tv = _numba.cuda.shared.array(Nc, _np.float32)
            else:
                refv = val[refr, :]
                tv = val[tr, :]
            loadid = _numba.cuda.threadIdx.x
            _numba.cuda.syncthreads()
            if shareval or sharetq or sharerefq:
                while loadid < _numba.int32(Nc):
                    if sharerefq:
                        refq0[loadid] = qx[refr, loadid]
                        refq1[loadid] = qy[refr, loadid]
                        refq2[loadid] = qz[refr, loadid]
                    if sharetq:
                        tq0[loadid] = qx[tr, loadid]
                        tq1[loadid] = qy[tr, loadid]
                        tq2[loadid] = qz[tr, loadid]
                    if shareval:
                        refv[loadid] = val[refr, loadid]
                        tv[loadid] = val[tr, loadid]
                    loadid += _numba.cuda.blockDim.x
            _numba.cuda.syncthreads()

            offsetc = _numba.int32(_numba.cuda.threadIdx.x - qmax)
            val = _numba.float32(0)
            dqold = _numba.cuda.threadIdx.x // 2
            dq = qmax + _numba.int32(1)
            for refc in range(_numba.int32(0), _numba.int32(Nc)):
                tc = _numba.int32(refc + offsetc)
                if 0 <= tc < Nc:
                    dq = _numba.int32(round(math.sqrt((refq0[refc] - tq0[tc]) ** 2 + (refq1[refc] - tq1[tc]) ** 2 + (refq2[refc] - tq2[tc]) ** 2)))
                    if dq <= qmax:
                        if dq != dqold:
                            _numba.cuda.atomic.add(out, (saveblock, dqold), _numba.float64(val))
                            dqold = dq
                            val = _numba.float32(0)
                        val += _numba.float32(refv[refc] * tv[tc])
            if dq < qmax:
                _numba.cuda.atomic.add(out, (saveblock, dq), _numba.float64(val))

        def reduce(vals):
            for i in range(1, vals.shape[0]):
                vals[0, _numba.cuda.threadIdx.x] += vals[i, _numba.cuda.threadIdx.x]
                vals[i, _numba.cuda.threadIdx.x] = 0

        def zero(vals):
            _numba.cuda.syncthreads()
            vals[_numba.cuda.blockIdx.x, _numba.cuda.threadIdx.x] = 0

        jkernel = _numba.cuda.jit("void(float32[:,:],float32[:,:],float32[:,:],float32[:,:],float64[:,:])", fastmath=True)(kernel)
        jkernel = jkernel[[(1, Nr, qmax), (int(2 * qmax + 1), 1, 1), stream]]
        jzero = _numba.cuda.jit("void(float64[:,:])")(zero)
        jzero = jzero[(doutput.shape[0]), (doutput.shape[1]), stream]
        jreduce = _numba.cuda.jit("void(float64[:,:])")(reduce)
        jreduce = jreduce[1, doutput.shape[1], stream]

        def corr(input):
            """
            Do the correlation
            """
            if stream is None:
                raise ValueError("already closed, use within with statement")
            with stream.auto_synchronize():
                dvals = cuda.to_device(input.astype(_np.float32), stream)
                jzero(doutput)
                jkernel(dqx, dqy, dqz, dvals, doutput)
                jreduce(doutput)
            return doutput[0, :qmax].copy_to_host(stream=stream).astype(_np.float64)

        yield corr
    finally:
        if stream is not None:
            stream.synchronize()
        stream = dqx = dqy = dqz = doutput = jkernel = jreduce = jzero = None


if __name__ == "__main__":
    qmax = 256
    z = 2000
    input = _np.ones((512, 512))
    with corrfunction(input.shape, z, qmax) as f:
        out = f(input)
        print(out.sum())
