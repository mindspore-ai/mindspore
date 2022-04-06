file(GLOB CUDA_LITE_KERNEL_SRC LIST_DIRECTORIES false
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cc
    ${CCSRC_DIR}/plugin/device/gpu/kernel/cuda_impl/cuda_class/*.cc
)

add_library(cuda_lite_kernel_mid OBJECT ${CUDA_LITE_KERNEL_SRC})
