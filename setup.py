from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize

import numpy

extensions = [
    Extension('fine.utils.im2col_cython', ['fine/utils/im2col/im2col_cython.pyx'],
              include_dirs=[numpy.get_include()]),
]

setup(
    name="deep-fine",
    version="0.0.5",
    description="Fine: a deep learning framework",
    long_description="Fine: a deep learning framework",
    long_description_content_type="text/markdown",
    author="Moussa Haidous",
    author_email="haidous.m@gmail.com",
    url="https://github.com/haidousm/fine",
    packages=["fine", "fine.utils",
              "fine.utils.im2col", "fine.activations",
              "fine.datasets", "fine.layers", "fine.loss",
              "fine.models", "fine.models.model_utils",
              "fine.models.model_utils.accuracy", "fine.optimizers"],
    install_requires=[
        'numpy',
        'cython',
        'tqdm',
        'requests',
        'opencv-python',
    ],
    ext_modules=cythonize(extensions),
)
