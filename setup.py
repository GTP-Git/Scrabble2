# setup.py

#Scrabble 26APR25 Cython V1


from setuptools import setup
from Cython.Build import cythonize
import numpy # Make sure numpy is installed (pip install numpy)

setup(
    ext_modules=cythonize("gaddag_cython.pyx"),
    include_dirs=[numpy.get_include()] # Include numpy headers if needed later
)
