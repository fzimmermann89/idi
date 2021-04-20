#!/usr/bin/env python
from os.path import join, exists, dirname, realpath, abspath, isdir
from os import environ, listdir, name as osname
from sys import prefix, path
import setuptools  # noqa # TODO


def configuration():
    from numpy.distutils.misc_util import Configuration
    from numpy.distutils.system_info import get_info, default_include_dirs, default_lib_dirs
    from collections import OrderedDict
    import numpy

    config = Configuration("idi", "")
    srcdir = join(dirname(realpath(__file__)), "idi")

    basedirs = list(
        OrderedDict.fromkeys(
            realpath(p)
            for p in [join(dirname(numpy.__file__), *(4 * [".."])), join(dirname(numpy.__file__), *(3 * [".."])), prefix]
            + [join(*p, *(2 * [".."])) for p in [p.split("site-packages")[:-1] for p in path] if p]
            + [join(*p, "..") for p in [p.split("site-packages")[:-1] for p in path] if p]
            + [join(p, "..") for p in environ["PATH"].split(":")]
        )
    )
    include_dirs = [srcdir]
    library_dirs = default_lib_dirs
    library_dirs.extend(join(b, "lib") for b in basedirs)
    library_dirs.extend(join(b, "lib64") for b in basedirs)
    library_dirs.extend(join(b, "libraries") for b in basedirs)
    library_dirs.extend(join(b, "Library/lib") for b in basedirs)
    library_dirs.extend(join(b, "Library/bin") for b in basedirs)

    include_dirs.extend(default_include_dirs)
    include_dirs.extend(join(b, "include") for b in basedirs)
    include_dirs.extend(join(b, "Library/include") for b in basedirs)

    include_dirs = [abspath(realpath(p)) for p in filter(isdir, include_dirs)]
    library_dirs = [abspath(realpath(p)) for p in filter(isdir, library_dirs)]

    files = ['mkl_intel_ilp64', 'mkl_core', 'mkl_intel_thread']

    if osname == 'nt':
        libextension = 'lib'
        libprefix = ''
        compileline = ' /DMKL_ILP64 /DNDEBUG /O2'
        print('basedirs', basedirs)
        print('include_dirs', include_dirs)
        print('library_dirs', library_dirs)

        linkline = "{paths}"
    else:
        libextension = 'a'
        libprefix = 'lib'
        linkline = "-Wl,{paths} -lpthread -lm"
        compileline = "-DNDEBUG -O3 -DMKL_ILP64"

    paths = []
    for f in files:
        for d in library_dirs:
            c = join(d, f'{libprefix}{f}.{libextension}')
            if exists(c):
                paths.append(c)
                break

    print(f'found {files} -> {paths}')

    try:
        from Cython.Build import cythonize

        sources = [join(srcdir, "reconstruction", "autocorrelate3.pyx")]
        have_cython = True
    except ImportError as e:
        have_cython = False
        sources = [join(srcdir, "reconstruction", "autocorrelate.c")]
        if not exists(sources[0]):
            raise ValueError(str(e) + ". " + "Cython is required to build the initial .c file.")
    config.add_extension(
        name="reconstruction.autocorrelate3",
        sources=sources,
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        extra_compile_args=compileline.split(" "),
        extra_link_args=linkline.format(paths=' '.join(paths)).split(" "),
    )
    if have_cython:
        config.ext_modules = cythonize(config.ext_modules)

    config.packages.append("idi")
    config.package_dir["idi"] = "./idi"
    config.packages.append("idi.simulation")
    config.package_dir["idi.simulation"] = "./idi/simulation"
    config.packages.append("idi.reconstruction")
    config.package_dir["idi.reconstruction"] = "./idi/reconstruction"
    config.packages.append("idi.util")
    config.package_dir["idi.util"] = "./idi/util"

    return config


def setup_package():
    from numpy.distutils.core import setup

    metadata = dict(
        maintainer="zimmf",
        description="idi simulation and reconstruction",
        platforms=["Linux", "Mac OS-X"],
        python_requires=">=3.6",
        install_requires=["numpy", "cython", "numba", "numexpr", "scipy", "jinja2", "mkl-service", "matplotlib", "h5py"],
        package_data={"": ["*.cu"]},
        scripts=["scripts/idi_sim.py", "scripts/idi_simrecon.py"],
        configuration=configuration,
        test_suite="tests",
    )
    setup(**metadata)

    return None


if __name__ == "__main__":
    setup_package()
