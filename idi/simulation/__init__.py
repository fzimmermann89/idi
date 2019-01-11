__all__=['cpu','simobj','cuda']
from . import cpu, simobj
try:
    from . import cuda
except ImportError as _e: 
    if "cuda" in _e.message:
        import warnings as _w
        _w.warn('cuda error. cuda simulation not imported')
    else: raise _e
