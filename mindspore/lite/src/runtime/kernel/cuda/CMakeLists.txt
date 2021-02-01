file(GLOB_RECURSE CUDA_KERNEL_SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/kernel/*.cc)
add_library(cuda_kernel_mid OBJECT ${CUDA_KERNEL_SRC})
add_dependencies(cuda_kernel_mid fbs_src)

