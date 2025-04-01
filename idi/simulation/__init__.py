__all__ = ["cpu", "simobj", "cuda", "common", "time", "cutime", "simple"]
import cupy as _cp
import numba.cuda as _numba_cuda

from . import cpu, simobj, time, simple
from .common import *

auto = None
autotime = None

try:
    from . import cuda
    from . import cutime
except ImportError as _e:
    if any(x in str(_e.args[0]).lower() for x in ["cuda", "libcu", "cupy"]):
        import warnings as _w

        _w.warn("cuda error. cuda time simulation not imported. is cuda available and paths set?")
        auto = cpu
        autotime = time
        _cuda = False
    else:
        print(_e)
except Exception as _e:
    print("Import exception:")
    print(_e)


if not _cp.cuda.is_available() or not _numba_cuda.is_available():
    _w.warn("cuda not available, using cpu version")
    auto = cpu
    autotime = time
else:
    auto = cuda
    autotime = cutime

del _cp
del _numba_cuda
