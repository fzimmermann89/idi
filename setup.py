#!/usr/bin/env python3

import os
import re
import sys
from pathlib import Path
from setuptools import setup, Extension, find_packages
from setuptools.command.sdist import sdist  # Python 3.12 compatible sdist
from Cython.Build import cythonize
import tomli
import numpy

# -----------------------------------------------------------------------------
# Heuristics for include and library directories


def find_include_and_lib_dirs():
    numpy_dir = Path(numpy.__file__).resolve()
    sys_prefix = Path(sys.prefix).resolve()

    candidate_dirs = [
        numpy_dir.parent.parent.parent.parent,
        numpy_dir.parent.parent.parent,
        sys_prefix,
    ]
    for p in sys.path:
        if p:
            parts = p.split("site-packages")
            if parts and parts[0]:
                candidate_dirs.append(Path(parts[0]) / ".." / "..")
                candidate_dirs.append(Path(parts[0]) / "..")
    for p in os.environ.get("PATH", "").split(os.pathsep):
        if p:
            candidate_dirs.append(Path(p) / "..")
    unique_basedirs = list(dict((str(p.resolve()), p.resolve()) for p in candidate_dirs).values())

    default_include_dirs, default_lib_dirs = [], []

    library_dirs = [Path(p) for p in default_lib_dirs]
    for b in unique_basedirs:
        library_dirs.append(b / "lib")
        library_dirs.append(b / "lib64")
        library_dirs.append(b / "libraries")
        library_dirs.append(b / "Library" / "lib")
        library_dirs.append(b / "Library" / "bin")
    include_dirs = [
        Path("idi"),
        Path("idi") / "reconstruction",
        Path("/usr/include"),
        Path("/usr/local/include"),
        Path("/usr/include/x86_64-linux-gnu"),
        Path("/usr/include/arm-linux-gnueabihf"),
        Path("/usr/include/i386-linux-gnu"),
        Path("/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include"),
        Path("/System/Library/Frameworks"),
    ]
    include_dirs.extend([Path(p) for p in default_include_dirs])

    for b in unique_basedirs:
        include_dirs.append(b / "include")
        include_dirs.append(b / "Library" / "include")
    library_dirs = [str(p.resolve()) for p in library_dirs if p.is_dir()]
    include_dirs = [str(p.resolve()) for p in include_dirs if p.is_dir()]
    basedirs = [str(p.resolve()) for p in unique_basedirs if p.is_dir()]

    return include_dirs, library_dirs, basedirs


# -----------------------------------------------------------------------------
# MKL detection heuristic remains mostly unchanged.


def get_mkl_info(include_dirs, library_dirs):
    found_mkl = False
    found_mkl_name = "mkl_rt"
    for d in library_dirs:
        dpath = Path(d)
        try:
            for f in dpath.iterdir():
                if f.name in {"mkl_rt.dll", "mkl_rt.so", "mkl_rt.dylib"}:
                    found_mkl = True
                    found_mkl_name = "mkl_rt"
                elif "mkl_rt.so." in f.name and not found_mkl:
                    found_mkl_name = ":" + f.name
                    found_mkl = True
            if found_mkl:
                break
        except Exception:
            continue
    libs = [found_mkl_name]
    if os.name != "nt":
        libs.extend(["pthread"])
    return [], libs


# -----------------------------------------------------------------------------
# Build the extension module.


def get_extension():
    src = Path("idi") / "reconstruction"
    sources = [str(src / "autocorrelate3.pyx")]
    print("Extension sources:", sources)
    heuristic_includes, heuristic_libs, _ = find_include_and_lib_dirs()
    include_dirs = [str(src), numpy.get_include()]
    include_dirs.extend(heuristic_includes)
    mkl_includes, mkl_libs = get_mkl_info(include_dirs, heuristic_libs)
    include_dirs.extend(mkl_includes)
    rpath = ["$ORIGIN/../../../../../../", "$ORIGIN/../../../../../", "$ORIGIN/../../../../", "$ORIGIN/../../../"]
    rpath = [f"-Wl,-rpath,{r}" for r in rpath]
    ext = Extension(
        name="idi.reconstruction.autocorrelate3",
        sources=sources,
        include_dirs=include_dirs,
        library_dirs=heuristic_libs,
        libraries=mkl_libs,
        extra_compile_args=["-DNDEBUG", "-O3", "-DMKL_ILP64"],
        extra_link_args=rpath,
    )
    return ext


def get_version(rel_path="idi/__init__.py"):
    version_file = Path(__file__).parent / rel_path
    content = version_file.read_text(encoding="utf-8")
    match = re.search(r'^__version__\s*=\s*[\'"]([^\'"]+)[\'"]', content, re.M)
    if match:
        return match.group(1)
    raise RuntimeError("Unable to find version string.")


# -----------------------------------------------------------------------------
# Read metadata from pyproject.toml.


def get_metadata():
    meta_file = Path(__file__).parent / "pyproject.toml"
    with meta_file.open("rb") as f:
        pyproject = tomli.load(f)
    project = pyproject["project"]
    return {
        "name": project["name"],
        "description": project["description"],
        "python_requires": project["requires-python"],
        "install_requires": project["dependencies"],
    }


# -----------------------------------------------------------------------------
# Setup the package.


def setup_package():
    ext = get_extension()
    ext_modules = cythonize(
        [ext],
        include_path=["idi/reconstruction"],
        compiler_directives={"language_level": "3"},
    )

    metadata = dict(
        version=get_version("idi/__init__.py"),
        packages=find_packages(),
        package_data={"": ["*.cu"]},
        cmdclass={"sdist": sdist},
        ext_modules=ext_modules,
    )
    metadata.update(get_metadata())
    setup(**metadata)


if __name__ == "__main__":
    setup_package()
