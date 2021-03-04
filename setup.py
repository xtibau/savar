from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
    name="savar",
    version="0.3",
    #  packages=["savar"],
    py_modules=['savar', 'savar.dim_methods', "savar.spatial_models", "savar.eval_tools", "savar.functions",
                "savar.model_generator", "savar.savar", "savar.c_functions"],
    ext_modules=cythonize("savar/*.pyx", compiler_directives={'language_level': "3"}),
    requires=['Cython', 'numpy', 'scipy', 'tigramite', 'matplotlib'],
    include_dirs=[numpy.get_include()])
