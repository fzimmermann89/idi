__all__ = [
    "hitcor",
    "hitcorrad",
    "cpucor",
    "cpucorrad",
    "cucor",
    "cucorrad",
    "cpusimple",
    "cusimple",
    "ft",
    "singleshotnorm",
]

import numba.cuda as _nbcuda
import warnings as _w

_w.filterwarnings("ignore", message="numpy.dtype size changed")
_w.filterwarnings("ignore", message="numpy.ufunc size changed")


from . import ( # noqa
    hitcor,
    hitcorrad,
    cpucor,
    cpucorrad,
    cpusimple,
    common,
    singleshotnorm,
    ft
)

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
    _w.warn("no cuda available", stacklevel=2)
    simple = cpusimple
    qcor = cpucor
    qcorrad = cpucorrad
