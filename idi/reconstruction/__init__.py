__all__ = ['direct', 'directrad', 'autocorrelate3', 'ft']

from . import direct, directrad

try:
    from . import autocorrelate3
except ImportError:
    _local = True
else:
    _local = False
if _local:
    try:
        import pyximport as _pyx
        from numpy.distutils.system_info import get_info as _getinfo
        from numpy import get_include as _np_get_include
        _mkl_inc = _getinfo('mkl').get('include_dirs')
        _np_inc = [_np_get_include()]
        _pyx.install(setup_args={'include_dirs': _mkl_inc + _np_inc })
    except ImportError:
        import warning as _w

        _w.warn("no cython!")
    else:
        from . import ft
else:
    from . import ft
