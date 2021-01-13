__all__ = ['cpu', 'simobj', 'cuda', 'common', 'time', 'cutime']
from . import cpu, simobj, common, time

try:
    from . import cuda
    auto = cuda
except ImportError as _e:
    if "cuda" in _e.args[0] or "libcu" in _e.args[0]:
        import warnings as _w
        _w.warn('cuda error. cuda simulation not imported. is cuda available and paths set?')
        auto = cpu
    else:
        raise _e
try:
    from . import cutime
    autotime = cutime
except (AttributeError,ImportError) as _e:
      if "cuda" in _e.args[0] or "libcu" in _e.args[0] or "split" in _e.args[0]:
        import warnings as _w
        _w.warn('cuda error. cuda time dependent simulation not imported. is cuda available and paths set?')
        autotime = time
  
