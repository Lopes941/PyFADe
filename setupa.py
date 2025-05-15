from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
import numpy

VERSION = '0.0.1'
DESCRIPTION = 'A failure and anomaly detection package in python'

cython_extensions = [
    Extension(
        name                ="pyfade.mprofile",
        sources             =["pyfade/mprofile.pyx"],
        include_dirs        =[numpy.get_include()],
        language            = 'c++',
        extra_compile_args  =["-O3"],
    ),
]


# Setup function for building the package
setup(
    name="pyfade",
    ext_modules=cythonize(cython_extensions),
    version=VERSION,
    author="Heitor Lopes",
    author_email="<hnlopes@unicamp.br>",
    description=DESCRIPTION,
    packages=find_packages(),
)
