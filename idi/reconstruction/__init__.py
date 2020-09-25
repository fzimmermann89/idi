__all__ = [ 'hitcor', 'hitcorrad', 'qcor', 'qcorrad', 'cucor', 'cucorrad', 'simple', 'ft']

import mkl as _mkl
_vml_threads=_mkl.domain_get_max_threads('vml')

from . import hitcor, hitcorrad, qcor, qcorrad, cucor, cucorrad, simple, common

try:
    from . import autocorrelate3
except ImportError as e:
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
        _pyx.install(setup_args={'include_dirs': _mkl_inc + _np_inc }, language_level=2)
    except ImportError:
        import warnings as _w

        _w.warn("no cython!")
    else:
        from . import ft
else:
    from . import ft

_mkl.domain_set_num_threads(_vml_threads,'vml') #numexpr messes with vml thread number