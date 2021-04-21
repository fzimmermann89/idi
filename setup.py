#!/usr/bin/env python
from os.path import join, exists, dirname, realpath, abspath, isdir
from os import environ, listdir
from os import name as osname
from sys import prefix, path
import setuptools  # noqa # TODO


def configuration():
    from numpy.distutils.misc_util import Configuration
    from numpy.distutils.system_info import (
        get_info,
        default_include_dirs,
        default_lib_dirs,
    )
    from collections import OrderedDict
    import numpy

    config = Configuration("idi", "")
    srcdir = join(dirname(realpath(__file__)), "idi")
    mkl_info = get_info("mkl")
    basedirs = list(
        OrderedDict.fromkeys(
            realpath(p)
            for p in [
                join(dirname(numpy.__file__), *(4 * [".."])),
                join(dirname(numpy.__file__), *(3 * [".."])),
                prefix,
            ]
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
    library_dirs.extend(join(b, "Library", "lib") for b in basedirs)
    library_dirs.extend(join(b, "Library", "bin") for b in basedirs)

    include_dirs.extend(default_include_dirs)
    include_dirs.extend(join(b, "include") for b in basedirs)
    include_dirs.extend(join(b, "Library", "include") for b in basedirs)

    # print("XXXXXXXXX")
    # print("basedirs", basedirs)
    # print("libdirs:", library_dirs)
    # print("includedirs:", include_dirs)
    # print("np", numpy.__file__)
    # print("env", environ)
    # print("path", path)
    # print("prefix", prefix)
    # print("XXXXXXXXXX")

    include_dirs = [abspath(realpath(p)) for p in filter(isdir, include_dirs)]
    library_dirs = [abspath(realpath(p)) for p in filter(isdir, library_dirs)]

    if mkl_info:
        include_dirs.extend(mkl_info.get("include_dirs"))
        libs = mkl_info.get("libraries", ["mkl_rt"])
    else:
        found_mkl = False
        found_mkl_name = "mkl_rt"
        for d in library_dirs:
            try:
                for f in listdir(d):
                    if f == "mkl_rt.dll" or f == "mkl_rt.so" or f == "mkl_rt.dylib":
                        found_mkl = True
                        found_mkl_name = "mkl_rt"
                    elif "mkl_rt.so." in f and not found_mkl:
                        found_mkl_name = ":" + f
                        found_mkl = True
                if found_mkl:
                    break
            except FileNotFoundError:
                continue
        libs = [found_mkl_name]
        if not osname == "nt":
            libs.extend(["pthread"])

    try:
        from Cython.Build import cythonize

        sources = [join(srcdir, "reconstruction", "autocorrelate3.pyx")]
        have_cython = True
    except ImportError as e:
        have_cython = False
        sources = [join(srcdir, "reconstruction", "autocorrelate.c")]
        if not exists(sources[0]):
            print("Cython is required to build ft autocorrelation")
    config.add_extension(
        name="reconstruction.autocorrelate3",
        sources=sources,
        libraries=libs,
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        extra_compile_args=["-DNDEBUG", "-O3", "-DMKL_ILP64"],
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


def get_version(rel_path):
    import os.path
    import codecs

    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), "r") as fp:
        for line in fp.read().splitlines():
            if line.startswith("__version__"):
                delim = '"' if '"' in line else "'"
                return line.split(delim)[1]
    raise RuntimeError("Unable to find version string.")


def setup_package():
    try:
        from numpy.distutils.core import setup
        config = configuration().todict()
    except ImportError:
        from setuptools import setup
        config = {"name": "idi"}

    metadata = dict(
        version=get_version("idi/__init__.py"),
        maintainer="zimmf",
        description="idi simulation and reconstruction",
        platforms=["Linux", "Mac OS-X"],
        python_requires=">=3.6",
        install_requires=[
            "numpy",
            "cython",
            "numba",
            "numexpr",
            "scipy",
            "jinja2",
            "mkl-service",
            "matplotlib",
            "h5py",
            "mkl",
            "mkl-include",
        ],
        package_data={"": ["*.cu"]},
        scripts=["scripts/idi_sim.py", "scripts/idi_simrecon.py"],
        test_suite="tests",
        **config
    )
    setup(**metadata)

    return None


if __name__ == "__main__":
    setup_package()
