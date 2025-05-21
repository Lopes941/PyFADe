cmake_minimum_required(VERSION 3.15...3.26)
project(pyfade LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(PYBIND11_FINDPYTHON ON)
find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
find_package(CUDAToolkit REQUIRED)


pybind11_add_module(mat_profile 
    src/pyfade/cpp/mprofile.cpp
    src/pyfade/bindings.cpp
    src/pyfade/cpp/cuda_funcs.cu)


target_link_libraries(mat_profile PRIVATE CUDA::cufft CUDA::cudart)
install(TARGETS mat_profile LIBRARY DESTINATION pyfade)
