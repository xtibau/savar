from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
    name="savar",
    version="0.4",
    #  packages=["savar"],
    py_modules=['savar', 'savar.dim_methods', "savar.old_spatial_models", "savar.eval_tools", "savar.functions",
                "savar.model_generator", "savar.savar"],
    requires=['numpy', 'scipy', 'tigramite', 'matplotlib'],
    ext_modules=cythonize(["./savar/c_functions.pyx","./savar/c_dim_methods.pyx"]),
    zip_safe=False,
    include_dirs=[numpy.get_include()])
