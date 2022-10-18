# cuda
find_package(CUDA)
file(GLOB_RECURSE CUDA_KERNEL_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cu
    ${CCSRC_DIR}/plugin/device/gpu/kernel/cuda_impl/cuda_ops/gather.cu
    ${CCSRC_DIR}/plugin/device/gpu/kernel/cuda_impl/cuda_ops/swish_impl.cu
    ${CCSRC_DIR}/plugin/device/gpu/kernel/cuda_impl/cuda_ops/cumsum_impl.cu
    ${CCSRC_DIR}/plugin/device/gpu/kernel/cuda_impl/cuda_ops/batchtospace_impl.cu
    ${CCSRC_DIR}/plugin/device/gpu/kernel/cuda_impl/cuda_ops/spacetobatch_impl.cu
    ${CCSRC_DIR}/plugin/device/gpu/kernel/cuda_impl/cuda_ops/select_impl.cu
    )

set_source_files_properties(${CUDA_KERNEL_SRC} PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGES} -std=c++14 -fPIC")
SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-std=c++14;-arch=sm_53)
cuda_add_library(cuda_kernel_mid STATIC ${CUDA_KERNEL_SRC})
