import os
from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize

VERSION = '0.0.1'
DESCRIPTION = 'A failure and anomaly detection package in python'

# Setting up

setup(
    name="pyfade",
    version=VERSION,
    author="Heitor Lopes",
    author_email="<hnlopes@unicamp.br>",
    description=DESCRIPTION,
    packages=find_packages(),
)

# import numpy
# import os
# here = os.path.abspath(os.path.dirname(__file__))
# source_dir = os.path.dirname(here) + "/pyfade"

# include_dirs = [here, source_dir, numpy.get_include()]
# for root, dirs, files in os.walk(here):
# 		include_dirs.append(root)

# ext_modules = [
#     Extension(
#         "pyfade",
#         ["pyfade/core/optimization.py"],
#         extra_compile_args=[],
# 		include_dirs=include_dirs,
#         define_macros=[("CYTHON_LIMITED_API", "1")],
#         py_limited_api=True,
#     )
# ]

# setup(ext_modules=cythonize(ext_modules, annotate=True,include_path=include_dirs),include_dirs=include_dirs)