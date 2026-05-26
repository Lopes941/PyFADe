cmake_minimum_required(VERSION 3.15)
project(pyfade LANGUAGES CXX)

# Enable CUDA if available
enable_language(CUDA OPTIONAL)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# ================================
# PYFADE SOURCE FILES
# ================================

file(GLOB BINDING_SOURCES src/pyfade/bindings.cpp)
file(GLOB CPU_SOURCES src/pyfade/cpp/cpu_funcs.cpp)
file(GLOB UTILS_SOURCES src/pyfade/cpp/utils.cpp)
file(GLOB DATASET_SOURCES src/pyfade/cpp/dataset.cpp)
file(GLOB FEATURES_SOURCES src/pyfade/cpp/features.cpp)
file(GLOB STATISTICS_SOURCES src/pyfade/cpp/statistics.cpp)
file(GLOB MAT_PROFILE_SOURCES src/pyfade/cpp/matrix_profile.cpp)
file(GLOB K_PROFILE_SOURCES src/pyfade/cpp/k_profile.cpp)

file(GLOB CUDA_SOURCES src/pyfade/cpp/cuda_funcs.cu)


set(SOURCE_FILES
    ${BINDING_SOURCES}
    ${CPU_SOURCES}
    ${UTILS_SOURCES}
    ${DATASET_SOURCES}
    ${FEATURES_SOURCES}
    ${STATISTICS_SOURCES}
    ${MAT_PROFILE_SOURCES}
    ${K_PROFILE_SOURCES}
    )


# ================================
# CUDA DETECTION
# ================================

find_package(CUDAToolkit QUIET)

if (DEFINED CUDAToolkit_ROOT)
    message(STATUS "CUDA found: Enabling GPU acceleration 1 ${CUDAToolkit_ROOT}")
    set(RESOLVED_CUDA_PATH "${CUDAToolkit_ROOT}")
    add_definitions(-DUSE_CUDA)
    set(HAS_CUDA TRUE)
elseif(DEFINED CUDA_PATH)
    message(STATUS "CUDA found: Enabling GPU acceleration 2 ${CUDA_PATH}")
    set(RESOLVED_CUDA_PATH "${CUDA_PATH}")
    add_definitions(-DUSE_CUDA)
    set(HAS_CUDA TRUE)
elseif (CUDAToolkit_FOUND)
    get_target_property(CUDART_LOCATION CUDA::cudart IMPORTED_LOCATION)
    get_filename_component(CUDA_LIB_DIR "${CUDART_LOCATION}" DIRECTORY)
    get_filename_component(CUDA_LIB_PARENT "${CUDA_LIB_DIR}" DIRECTORY)
    get_filename_component(CUDA_TOOLKIT_ROOT "${CUDA_LIB_PARENT}" DIRECTORY)
    message(STATUS "CUDA found: Enabling GPU acceleration 3 ${CUDA_TOOLKIT_ROOT}")
    set(RESOLVED_CUDA_PATH "${CUDA_TOOLKIT_ROOT}")
    add_definitions(-DUSE_CUDA)
    set(HAS_CUDA TRUE)
else()
    message(STATUS "CUDA not found: Building CPU-only version")
    set(RESOLVED_CUDA_PATH "")
    set(HAS_CUDA FALSE)
endif()

if (HAS_CUDA)
    list(APPEND SOURCE_FILES ${CUDA_SOURCES})
endif()

# ================================
# CONFIG GENERATION 
# ================================

# Write the path to a Python file inside the package
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/cuda_config.py"
     "cuda_path = r'${RESOLVED_CUDA_PATH}'\n")

     
# Install the config file into the module directory
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/cuda_config.py"
        DESTINATION pyfade)


# =================================
# Install Python extension + stubs
# =================================

pybind11_add_module(core ${SOURCE_FILES})

target_include_directories(core PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)

# macOS specific
if(APPLE)
    set_target_properties(core PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
endif()

if (HAS_CUDA)
    target_link_libraries(core PRIVATE CUDA::cufft CUDA::cudart)
    set_target_properties(core PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()


install(TARGETS core 
    LIBRARY DESTINATION pyfade
    )
