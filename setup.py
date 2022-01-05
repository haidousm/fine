import sys
import numpy

from setuptools import setup, Extension
from Cython.Build import cythonize

number_of_arguments = len(sys.argv)
version_parameter = sys.argv[-1]
version = version_parameter.split("=")[1]
sys.argv = sys.argv[0:number_of_arguments - 1]

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

extensions = [
    Extension('fine.utils.im2col_cython', ['fine/utils/im2col/im2col_cython.pyx'],
              include_dirs=[numpy.get_include()]),
]

setup(
    name="deep-fine",
    version=version,
    description="an artificial neural network framework built from scratch using just Python and Numpy",
    author="Moussa Haidous",
    author_email="haidous.m@gmail.com",
    url="https://github.com/haidousm/fine",
    packages=["fine", "fine.utils",
              "fine.utils.im2col", "fine.activations",
              "fine.datasets", "fine.layers", "fine.loss",
              "fine.models", "fine.models.model_utils",
              "fine.models.model_utils.accuracy", "fine.optimizers"],
    install_requires=requirements,
    ext_modules=cythonize(extensions),
)
