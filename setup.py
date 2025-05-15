import os
import subprocess
from setuptools import setup, Extension, find_packages
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import numpy

VERSION = '0.0.1'
DESCRIPTION = 'A failure and anomaly detection package in python'

# Ensure CUDA_PATH is set
cuda_path = os.environ.get("CUDA_PATH", "")


if not cuda_path:
    raise EnvironmentError("CUDA_PATH is not set. Ensure you're using a Conda environment with CUDA installed.")

# Directories for CUDA libraries and includes
cuda_lib_dir = os.path.join(cuda_path, "lib", "x64")
cuda_include_dir = os.path.join(cuda_path, "include")
libraries = ["cuda", "cufft", "cudart"]
nvcc_loc = os.path.join(cuda_path, "bin", "nvcc.exe")

# Ensure nvcc is available
if not os.path.exists(nvcc_loc):
    raise EnvironmentError(f"nvcc not found at {nvcc_loc}. Ensure CUDA is installed properly.")

# Compile CUDA .cu file to .dll
def compile_cuda():
    # Check if the .dll already exists to avoid recompiling unnecessarily
   # if not os.path.exists("pyfade/core/cuda_funcs.cubin"):
    print("Compiling CUDA code...")
    subprocess.check_call([nvcc_loc, "-cubin", "-o", "pyfade/core/cuda_funcs.cubin", "-arch=sm_75", "pyfade/core/cuda_funcs.cu", "-lcudart", "-lcufft"])
   # else:
    #    pass

# Compile CUDA code before running the setup
#compile_cuda()


# Define Cython extensions
cython_extensions = [
    Extension(
        name                ="pyfade.core.mprofile",
        sources             =["pyfade/core/mprofile.pyx" + "pyfade/core/cuda_funcs.cu"],
        include_dirs        =[numpy.get_include()],
        language            = 'c++',
        extra_compile_args  =["-O3"],
    ),
]

# Combine both CUDA and Cython extensions
ext_modules = cythonize(cython_extensions, annotate=True)

package_data = {
    'pyfade': ['core/cuda_funcs.cubin'],  # Include your .cubin file in the package
}

# Setup function for building the package
setup(
    name="pyfade",
    version=VERSION,
    author="Heitor Lopes",
    author_email="<hnlopes@unicamp.br>",
    description=DESCRIPTION,
    packages=find_packages(),
    package_data=package_data,
    ext_modules=ext_modules,
    include_package_data=True,
    install_requires=["cython", "numpy"],  # Add other dependencies as necessary
)
