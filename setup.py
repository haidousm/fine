from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext as _build_ext
from Cython.Build import cythonize


class build_ext(_build_ext):
    def finalize_options(self):
        _build_ext.finalize_options(self)
        # Prevent numpy from thinking it is still in its setup process:
        __builtins__.__NUMPY_SETUP__ = False
        import numpy
        self.include_dirs.append(numpy.get_include())


with open('requirements.txt') as f:
    requirements = f.read().splitlines()

extensions = [
    Extension('fine.utils.im2col_cython', ['fine/utils/im2col/im2col_cython.pyx']),
]

setup(
    name="deep-fine",
    version="1.0.1",
    description="an artificial neural network framework built from scratch using just Python and Numpy",
    author="Moussa Haidous",
    author_email="haidous.m@gmail.com",
    url="https://github.com/haidousm/fine",
    packages=["fine", "fine.utils",
              "fine.utils.im2col", "fine.activations",
              "fine.datasets", "fine.layers", "fine.loss",
              "fine.models", "fine.models.model_utils",
              "fine.models.model_utils.accuracy", "fine.optimizers"],
    cmdclass={'build_ext': build_ext},
    install_requires=requirements,
    ext_modules=cythonize(extensions),
)
