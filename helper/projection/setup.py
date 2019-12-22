from distutils.core import setup
from Cython.Build import cythonize
import numpy as np

setup(
    include_dirs = [np.get_include()],
    ext_modules = cythonize("./helper/projection/projection.pyx",
            compiler_directives={'language_level' : "3"},
            )
)