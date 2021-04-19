__all__ = ['hitcor', 'hitcorrad', 'cpucor', 'cpucorrad', 'cucor', 'cucorrad', 'cpusimple', 'cusimple', 'ft']

import mkl as _mkl
import numba.cuda as _nbcuda
import warnings as _w

_w.filterwarnings("ignore", message="numpy.dtype size changed")
_w.filterwarnings("ignore", message="numpy.ufunc size changed")

_vml_threads = _mkl.domain_get_max_threads('vml')

from . import hitcor, hitcorrad, cpucor, cpucorrad, cpusimple, common  # noqa

# mkl fft
try:
    from . import autocorrelate3  # noqa

    _local = False
except ImportError as e:
    _w.warn('trying local compilation' + str(e))
    _local = True

    try:
        import pyximport as _pyx
        from sys import prefix as _prefix
        from os.path import join as _join
        from numpy.distutils.system_info import get_info as _getinfo, default_include_dirs as _defaultincludedirs
        from numpy import get_include as _np_get_include

        _incs = _defaultincludedirs
        _incs.append(_np_get_include())
        _incs.append(_join(_prefix, 'include'))
        _mklinc = _getinfo('mkl').get('include_dirs')
        if _mklinc:
            _incs.extend(_mklinc)
        _pyx.install(setup_args={'include_dirs': _incs}, language_level=2)
    except Exception as e:
        _w.warn("no mkl autocorrelation")

from . import ft  # noqa

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
