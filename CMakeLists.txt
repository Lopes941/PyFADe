cmake_minimum_required(VERSION 3.18)
project(pyfade LANGUAGES C CXX CUDA)

enable_language(CUDA)

# Let scikit-build-core handle Python & NumPy setup
find_package(Python3 REQUIRED COMPONENTS Development.Module)
find_package(NumPy REQUIRED)

# Optional: any C++/CUDA flags
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 14)
