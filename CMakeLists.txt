cmake_minimum_required(VERSION 3.19)
project(kernel_forge LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

set(CUDA_COMPUTE_CAPABILITY 80 CACHE STRING "GPU compute capability (e.g. 80 for A100, 89 for L4, 90 for H100)")

find_package(CUDAToolkit REQUIRED)

include_directories(${PROJECT_SOURCE_DIR})

set(EXERCISES
    matmul
    reduction
    cross_entropy
    flash_attention
    quantized_gemm
    moe_dispatch
)

foreach(exercise ${EXERCISES})
    add_executable(${exercise} runners/${exercise}.cu)
    set_target_properties(${exercise} PROPERTIES
        CUDA_ARCHITECTURES ${CUDA_COMPUTE_CAPABILITY}
        CUDA_SEPARABLE_COMPILATION ON
    )
    target_link_libraries(${exercise} PRIVATE CUDA::cudart CUDA::cublas)
endforeach()
