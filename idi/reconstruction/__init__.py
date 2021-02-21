__all__ = ['hitcor', 'hitcorrad', 'cpucor', 'cpucorrad', 'cucor', 'cucorrad', 'cpusimple', 'cusimple', 'ft']

import mkl as _mkl
import numba.cuda as _nbcuda
import warnings as _w

_vml_threads = _mkl.domain_get_max_threads('vml')

from . import hitcor, hitcorrad, cpucor, cpucorrad, cpusimple, common  # noqa

# mkl fft
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
        _pyx.install(setup_args={'include_dirs': _mkl_inc + _np_inc}, language_level=2)
    except ImportError:
        _w.warn("ft autocorrelate")
    else:
        from . import ft
else:
    from . import ft

# cuda
_cuda = _nbcuda.is_available()
if _cuda:
    try:
        from . import cucor, cucorrad, cusimple

        simple = cusimple
        qcor = cucor
        qcorrad = cucorrad

    except Exception as e:
        print(e)
        _cuda = False
if not _cuda:
    _w.warn("no cuda available")
    simple = cpusimple
    qcor = cpucor
    qcorrad = cpucorrad


_mkl.domain_set_num_threads(_vml_threads, 'vml')  # numexpr messes with vml thread number
