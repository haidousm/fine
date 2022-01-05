from setuptools import setup, Extension
from Cython.Build import cythonize

import numpy

extensions = [
    Extension('utils.im2col_cython', ['utils/im2col/im2col_cython.pyx'],
              include_dirs=[numpy.get_include()]),
]

setup(
    name="vdeep-fine",
    version="0.0.1",
    description="Fine: A deep learning framework",
    author="Moussa Haidous",
    author_email="haidous.m@gmail.com",
    url="https://github.com/haidousm/fine",
    packages=['fine'],
    ext_modules=cythonize(extensions),
)
