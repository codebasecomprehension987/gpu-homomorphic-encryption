cmake_minimum_required(VERSION 3.18)
project(FHE_CUDA LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89 90)

find_package(CUDAToolkit REQUIRED)

# Compiler flags for maximum performance
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -use_fast_math")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xptxas -v")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --generate-line-info")

include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/kernels)

# Main library
set(FHE_SOURCES
    src/bigint.cu
    src/ntt.cu
    src/polynomial.cu
    src/rns.cu
    src/fhe.cu
    kernels/ntt_kernels.cu
)

add_library(fhe_cuda STATIC ${FHE_SOURCES})
target_link_libraries(fhe_cuda CUDA::cudart CUDA::cufft)

# Test executable
add_executable(test_fhe tests/test_fhe.cu)
target_link_libraries(test_fhe fhe_cuda)

# Set compute capability specific optimizations
set_target_properties(fhe_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_PTX_COMPILATION ON
)
