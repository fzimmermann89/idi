__all__ = ['cpu', 'simobj', 'cuda', 'common', 'time', 'cutime', 'simple']
from . import cpu, simobj, time, simple
from .common import *

try:
    from . import cuda
    auto = cuda
except ImportError as _e:
    if any(x in str(_e.args[0]).lower() for x in ["cuda", "libcu", "cupy"]):
        import warnings as _w
        _w.warn('cuda error. cuda time dependent simulation not imported. is cuda available and paths set?')
        auto = cpu
    else:
        print(_e)
except Exception as _e:
    print('Import exception on cuda:')
    print(_e)
try:
    from . import cutime
    autotime = cutime
except (AttributeError, ImportError) as _e:
    if any(x in str(_e.args[0]).lower() for x in ["cuda", "libcu", "cupy"]):
        import warnings as _w
        _w.warn('cuda error. cuda time dependent simulation not imported. is cuda available and paths set?')
        autotime = time
    else:
        print(_e)
except Exception as _e:
    print('Import exception on cutime:')
    print(_e)
