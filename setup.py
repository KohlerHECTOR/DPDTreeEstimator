"""Build script for the dpdt._core C++ extension (see pyproject.toml)."""

import sys

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

extra_compile_args = [] if sys.platform == "win32" else ["-O3"]

ext_modules = [
    Pybind11Extension(
        "dpdt._core",
        ["dpdt/_core.cpp"],
        cxx_std=17,
        extra_compile_args=extra_compile_args,
    ),
]

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
