from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize("brax_trainer/c_gae.pyx"),
    include_dirs=[numpy.get_include()],
)
