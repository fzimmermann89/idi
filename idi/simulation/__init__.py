__all__ = ["cpu", "simobj", "cuda", "common", "time", "cutime", "simple"]
import warnings as _w

from . import cpu, simobj, time, simple
from .common import *

_cuda = False

try:
    import cupy as _cp
    import numba.cuda as _numba_cuda

    if _cp.cuda.is_available() and _numba_cuda.is_available():
        from . import cuda
        from . import cutime
        auto = cuda
        autotime = cutime
        _cuda = True

except ImportError as _e:
    if not any(x in str(_e.args[0]).lower() for x in ["cuda", "libcu", "cupy"]):
        print(_e)
except Exception as _e:
    print("Import exception:")
    print(_e)


if not _cuda:
    _w.warn("cuda error. cuda time simulation not imported. is cuda available and paths set?")
    auto = cpu
    autotime = time
