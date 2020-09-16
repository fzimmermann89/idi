from __future__ import print_function
import sys
from os.path import join, exists, dirname, realpath
from os import getcwd
import IPython


def configuration():
    from numpy.distutils.misc_util import Configuration
    from numpy.distutils.system_info import get_info

    config = Configuration('idi', '')
    srcdir = join(dirname(realpath(__file__)), 'idi')
    mkl_info = get_info('mkl')
    libs = mkl_info.get('libraries', ['mkl_rt'])
    include_dirs = mkl_info.get('include_dirs') + [srcdir]
    try:
        from Cython.Build import cythonize

        sources = [join(srcdir, 'reconstruction', 'autocorrelate3.pyx')]
        have_cython = True
    except ImportError as e:
        have_cython = False
        sources = [join(srcdir, 'reconstruction', 'autocorrelate.c')]
        if not exists(sources[0]):
            raise ValueError(str(e) + '. ' + 'Cython is required to build the initial .c file.')
    config.add_extension(
        name='reconstruction.autocorrelate3',
        sources=sources,
        libraries=libs,
        include_dirs=include_dirs,
        extra_compile_args=['-DNDEBUG'],
    )
    if have_cython:
        config.ext_modules = cythonize(config.ext_modules)

    config.packages.append('idi')
    config.package_dir['idi'] = './idi'
    config.packages.append('idi.simulation')
    config.package_dir['idi.simulation'] = './idi/simulation'
    config.packages.append('idi.reconstruction')
    config.package_dir['idi.reconstruction'] = './idi/reconstruction'
    config.packages.append('idi.util')
    config.package_dir['idi.util'] = './idi/util'
    return config


def setup_package():
    from numpy.distutils.core import setup

    metadata = dict(
        maintainer='zimmf',
        description='idi simulation and reconstruction',
        platforms=['Linux', 'Mac OS-X'],
        python_requires='>=3.6',
        install_requires=['numpy', 'cython', 'numba', 'numexpr', 'scipy', 'jinja', 'mkl-devel'],
        scripts=['idi_sim.py', 'idi_simrecon.py', 'idi_simrecon_normalize.py'],
        configuration=configuration,
    )
    setup(**metadata)

    return None


if __name__ == '__main__':
    setup_package()
