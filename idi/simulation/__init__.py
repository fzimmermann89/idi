__all__ = ['cpu', 'simobj', 'cuda', 'common']
from . import cpu, simobj, common

try:
    from . import cuda
    auto = cuda
except ImportError as _e:
    if "cuda" in _e.args[0] or "libcu" in _e.args[0]:
        import warnings as _w
        _w.warn('cuda error. cuda simulation not imported. is cuda available and nvcc path?')
        auto = cpu
    else:
        raise _e
