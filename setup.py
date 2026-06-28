from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    name="savar",
    version="0.4",
    packages=["savar"],
    install_requires=['numpy', 'scipy', 'tigramite', 'matplotlib', 'scikit-learn', 'cython'],
    ext_modules=cythonize(["./savar/c_functions.pyx", "./savar/c_dim_methods.pyx"]),
    zip_safe=False,
    include_dirs=[numpy.get_include()])
