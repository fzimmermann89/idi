__all__ = ['cpu', 'simobj', 'cuda', 'common', 'time', 'cutime', 'simple']

try:
    import mkl as _mkl

    _vml_threads = _mkl.domain_get_max_threads('vml')
except ImportError:
    _mkl = None

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

if _mkl is not None:
    _mkl.domain_set_num_threads(_vml_threads, 'vml')  # numexpr messes with vml thread number
