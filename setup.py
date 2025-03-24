#!/usr/bin/env python
from os.path import join, exists, dirname, realpath, abspath, isdir
from os import environ, listdir
from os import name as osname
from sys import prefix, path
import setuptools  # noqa # TODO
from distutils.command.sdist import sdist


def path(x):
    """for python 3.8 on github actions"""
    try:
        x = realpath(x)
        if not exists(x):
            return None
        return abspath(x)
    except:
        return None


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
    srcdir = "./idi"
    mkl_info = get_info("mkl")
    basedirs = list(
        OrderedDict.fromkeys(
            path(p)
            for p in [
                join(dirname(numpy.__file__), *(4 * [".."])),
                join(dirname(numpy.__file__), *(3 * [".."])),
                prefix,
            ]
            + [join(*p, *(2 * [".."])) for p in [p.split("site-packages")[:-1] for p in path] if p]
            + [join(*p, "..") for p in [p.split("site-packages")[:-1] for p in path] if p]
            + [join(p, "..") for p in environ["PATH"].split(":")]
            if path(p)
        )
    )
    include_dirs = [
        srcdir,
        "/usr/include",
        "/usr/local/include",
        "/usr/include/x86_64-linux-gnu",
        "/usr/include/i386-linux-gnu",
        "/usr/include/arm-linux-gnueabihf",
        "/usr/include/clang",
        "/usr/include/c++",
        "/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include",
        "/System/Library/Frameworks",
        "C:/MinGW/include",
        "C:/msys64/mingw32/include",
        "C:/msys64/mingw64/include",
        "C:/Program Files (x86)/Microsoft Visual Studio/VC/Tools/MSVC/include",
        "C:/Windows Kits/10/Include/um",
        "C:/Windows Kits/10/Include/shared",
        "/opt/include",
    ]
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
        if not exists(sources[0]):
            print("pyx missing")
            raise FileNotFoundError
    except (ImportError, FileNotFoundError) as e:
        have_cython = False
        sources = [join(srcdir, "reconstruction", "autocorrelate3.c")]
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
        config.ext_modules = cythonize(
            config.ext_modules,
            include_path=[abspath(realpath(join(srcdir, "reconstruction")))],
        )
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


def get_metadata():
    """Extracts project metadata from pyproject.toml."""
    import os
    import tomli

    pyproject_path = os.path.join(os.path.dirname(__file__), "pyproject.toml")
    with open(pyproject_path, "rb") as f:
        pyproject = tomli.load(f)

    project = pyproject["project"]
    return {
        "name": project["name"],
        "description": project["description"],
        "maintainer": project["authors"][0]["name"],
        "maintainer_email": project["authors"][0]["email"],
        "python_requires": project["requires-python"],
        "install_requires": project["dependencies"],
    }


def setup_package():
    try:
        from numpy.distutils.core import setup

        config = configuration().todict()
    except ImportError:
        from setuptools import setup

        config = {}

    metadata = dict(
        version=get_version("idi/__init__.py"),
        package_data={"": ["*.cu"]},
        scripts=["scripts/idi_sim.py", "scripts/idi_simrecon.py"],
        test_suite="tests",
        cmdclass={"sdist": sdist},
        packages=["idi"],
        package_dir={"": "."},
    )
    metadata.update(config)
    metadata.update(get_metadata())

    setup(**metadata)

    return None


if __name__ == "__main__":
    setup_package()
