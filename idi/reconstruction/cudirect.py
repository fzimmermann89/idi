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
    dzd = None
    dvals = None
    doutput = None
    jkernel = None

    try:
        numba.cuda.select_device(0)
        stream = numba.cuda.stream()
        Nr, Nc = int(shape[0]), int(shape[1])
        if not 32 <= maxq <= 1024:
            warnings.warn('maxq will be clamped between 32 and 1024')
        qmax = numba.int32(max(32, min(1024, maxq, Nr, Nc)))
        y, x = np.meshgrid(np.arange(Nc, dtype=np.float64), np.arange(Nr, dtype=np.float64))
        xcenter = numba.float32(xcenter or Nr / 2)
        ycenter = numba.float32(xcenter or Nc / 2)

        x -= xcenter
        y -= ycenter
        d = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        zd = np.array((z / d), np.float32, order='C')
        maxdqz = numba.int32(math.ceil(z * (np.max(zd) - np.min(zd))))
        cz = numba.float32(z)

        with stream.auto_synchronize():
            dzd = numba.cuda.to_device(zd, stream)
            doutput = numba.cuda.device_array((2 * maxdqz + 2, 2 * qmax, 2 * qmax), np.float32, stream=stream)

        def corr(input, fkernel, fassemble):
            with stream.auto_synchronize():
                dvals = numba.cuda.to_device(input.astype(np.float32), stream)
                doutput[:, :, :] = 0
                fkernel(dzd, dvals, doutput)
                fassemble(doutput)
                return doutput.copy_to_host(stream=stream)

        def kernel(zd, val, out):
            refr = numba.int32(numba.cuda.blockIdx.y)
            offsetr = numba.int32(numba.cuda.blockIdx.z)
            # offsetc=numba.cuda.threadIdx.x-qmax
            refzd = numba.cuda.shared.array((Nc), numba.float32)
            refv = numba.cuda.shared.array((Nc), numba.float32)
            tzd = numba.cuda.shared.array((Nc), numba.float32)
            tv = numba.cuda.shared.array((Nc), numba.float32)
            tr = numba.int32(refr + offsetr)
            if not (0 <= tr < Nr and 0 <= refr < Nr):
                return

            loadid = numba.int32(numba.cuda.threadIdx.x)
            numba.cuda.syncthreads()
            while loadid < Nc:
                refzd[loadid] = zd[refr, loadid]
                refv[loadid] = val[refr, loadid]
                tzd[loadid] = zd[tr, loadid]
                tv[loadid] = val[tr, loadid]
                loadid += numba.cuda.blockDim.x
            numba.cuda.syncthreads()
            tr2 = numba.float32(tr) - xcenter
            refr2 = numba.float32(refr) - xcenter
            offsetc = -numba.int32(numba.cuda.threadIdx.x)
            while offsetc < qmax:
                for refc in range(0, Nc):
                    tc = numba.int32(refc + offsetc)
                    if 0 <= tc < Nc:
                        dqx = qmax + numba.int32(round(tr2 * tzd[tc] - refr2 * refzd[refc]))
                        dqy = qmax + numba.int32(round((numba.float32(tc) - ycenter) * tzd[tc] - (numba.float32(refc) - ycenter) * refzd[refc]))
                        dqz = maxdqz + numba.int32(round(cz * (refzd[refc] - tzd[tc])))
                        numba.cuda.atomic.add(out, (dqz, dqx, dqy), refv[refc] * tv[tc])
                offsetc += numba.int32(numba.cuda.blockDim.x)

        def assemble(vals):
            idx, idy, idz = numba.cuda.grid(3)
            idy2 = vals.shape[1] - idy - 1
            tmp = vals[idx, idy, idz]
            vals[idx, idy, idz] += vals[idx, idy2, idz]
            vals[idx, idy2, idz] += tmp

        jkernel = numba.cuda.jit("void(float32[:,:],float32[:,:],float32[:,:,:])", fastmath=True)(kernel)
        # print(jkernel.inspect_asm())
        jkernel = jkernel[(1, Nr, qmax), (qmax, 1, 1), stream]
        jassemble = numba.cuda.jit("void(float32[:,:,:])", fastmath=True)(assemble)
        jassemble = jassemble[(doutput.shape[0], 1, doutput.shape[2]), (1, qmax, 1), stream]
        yield functools.partial(corr, fkernel=jkernel, fassemble=jassemble)
    finally:
        if stream is not None:
            stream.synchronize()
        jkernel = None
        stream = None
        dzd = None
        dvals = None
        doutput = None
        jkernel = None
        gc.collect()


if __name__ == '__main__':
    qmax = 256
    z = 2000
    input = np.ones((512, 512))
    with corrfunction(input.shape, z, qmax) as f:
        out = f(input)
        print(out.sum())
