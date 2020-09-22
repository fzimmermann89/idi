import functools
import numba
import numpy as np
import math
import contextlib
import warnings
import gc


@contextlib.contextmanager
def corrfunction(shape, z, maxq, xcenter=None, ycenter=None):
    stream = None
    saveblocks = numba.int32(32)
    try:
        if not 15 <= maxq <= 511:
            warnings.warn('maxq will be clamped between 15 and 511')
        numba.cuda.select_device(0)
        stream = numba.cuda.stream()
        qmax = numba.int32(max(15, min(511, maxq)))
        Nr, Nc = int(shape[0]), int(shape[1])
        y, x = np.meshgrid(np.arange(Nc, dtype=np.float64), np.arange(Nr, dtype=np.float64))
        xcenter = numba.float32(xcenter or Nr / 2)
        ycenter = numba.float32(xcenter or Nc / 2)
        x -= xcenter
        y -= ycenter
        d = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        qx, qy, qz = ((k * z / d).astype(np.float32) for k in (x, y, z))

        with stream.auto_synchronize():
            dqx = numba.cuda.to_device(qx, stream)
            dqy = numba.cuda.to_device(qy, stream)
            dqz = numba.cuda.to_device(qz, stream)
            doutput = numba.cuda.device_array((saveblocks, qmax + 1), np.float64, stream=stream)

        def corr(input, fkernel, fzero, freduce):
            with stream.auto_synchronize():
                dvals = numba.cuda.to_device(input.astype(np.float32), stream)
                fzero(doutput)
                fkernel(dqx, dqy, dqz, dvals, doutput)
                freduce(doutput)
            return doutput[0, :qmax].copy_to_host(stream=stream)

        def kernel(qx, qy, qz, val, out):
            refr = numba.cuda.blockIdx.y
            offsetr = numba.cuda.blockIdx.z

            saveblock = numba.int32(numba.cuda.threadIdx.x % saveblocks)
            refq0, refq1, refq2 = numba.cuda.shared.array((Nc), np.float32), numba.cuda.shared.array((Nc), np.float32), numba.cuda.shared.array((Nc), np.float32)
            refv = numba.cuda.shared.array((Nc), np.float32)
            tq0, tq1, tq2 = numba.cuda.shared.array((Nc), np.float32), numba.cuda.shared.array((Nc), np.float32), numba.cuda.shared.array((Nc), np.float32)
            tv = numba.cuda.shared.array((Nc), np.float32)
            tr = numba.int32(refr + offsetr)
            if not (0 <= tr < Nr and 0 <= refr < Nr):
                return
            loadid = numba.cuda.threadIdx.x
            numba.cuda.syncthreads()
            while loadid < numba.int32(Nc):
                refq0[loadid] = qx[refr, loadid]
                refq1[loadid] = qy[refr, loadid]
                refq2[loadid] = qz[refr, loadid]
                refv[loadid] = val[refr, loadid]
                tq0[loadid] = qx[tr, loadid]
                tq1[loadid] = qy[tr, loadid]
                tq2[loadid] = qz[tr, loadid]
                tv[loadid] = val[tr, loadid]
                loadid += numba.cuda.blockDim.x
            numba.cuda.syncthreads()

            offsetc = numba.int32(numba.cuda.threadIdx.x - qmax)
            val = numba.float32(0)
            dqold = numba.cuda.threadIdx.x // 2
            dq = qmax + numba.int32(1)
            for refc in range(numba.int32(0), numba.int32(Nc)):
                tc = numba.int32(refc + offsetc)
                if 0 <= tc < Nc:
                    dq = numba.int32(round(math.sqrt((refq0[refc] - tq0[tc]) ** 2 + (refq1[refc] - tq1[tc]) ** 2 + (refq2[refc] - tq2[tc]) ** 2)))
                    if dq <= qmax:
                        if dq != dqold:
                            numba.cuda.atomic.add(out, (saveblock, dqold), numba.float64(val))
                            dqold = dq
                            val = numba.float32(0)
                        val += numba.float32(refv[refc] * tv[tc])
            if dq < qmax:
                numba.cuda.atomic.add(out, (saveblock, dq), numba.float64(val))

        def reduce(vals):
            for i in range(1, vals.shape[0]):
                vals[0, numba.cuda.threadIdx.x] += vals[i, numba.cuda.threadIdx.x]
                vals[i, numba.cuda.threadIdx.x] = 0

        def zero(vals):
            numba.cuda.syncthreads()
            vals[numba.cuda.blockIdx.x, numba.cuda.threadIdx.x] = 0

        jkernel = numba.cuda.jit("void(float32[:,:],float32[:,:],float32[:,:],float32[:,:],float64[:,:])", fastmath=True)(kernel)
        # print(jkernel.inspect_asm())
        jkernel = jkernel[[(1, Nr, qmax), (int(2 * qmax + 1), 1, 1), stream]]
        jzero = numba.cuda.jit("void(float64[:,:])", fastmath=True)(zero)[(doutput.shape[0]), (doutput.shape[1]), stream]
        jreduce = numba.cuda.jit("void(float64[:,:])", fastmath=True)(reduce)[(1), doutput.shape[1], stream]
        yield functools.partial(corr, fkernel=jkernel, freduce=jreduce, fzero=jzero)
    finally:
        if stream is not None:
            stream.synchronize()
        dqx = None
        dqy = None
        dqz = None
        doutput = None
        dvals = None
        jkernel = None
        jzero = None
        jreduce = None
        gc.collect()


if __name__ == '__main__':
    qmax = 256
    z = 2000
    input = np.ones((512, 512))
    with corrfunction(input.shape, z, qmax) as f:
        out = f(input)
        print(out.sum())
