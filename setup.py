from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
    name="savar",
    version="0.4",
    #  packages=["savar"],
    py_modules=['savar', 'savar.dim_methods', "savar.spatial_models", "savar.eval_tools", "savar.functions",
                "savar.model_generator", "savar.savar"],
    requires=['numpy', 'scipy', 'tigramite', 'matplotlib'],
    include_dirs=[numpy.get_include()])
