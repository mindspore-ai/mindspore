#cuda
find_package(CUDA)
add_compile_definitions(ENABLE_GPU)
file(GLOB_RECURSE CUDA_KERNEL_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cu
    ${CCSRC_DIR}/plugin/device/gpu/kernel/cuda_impl/cuda_ops/gather.cu
    ${CCSRC_DIR}/plugin/device/gpu/kernel/cuda_impl/cuda_ops/swish_impl.cu
    ${CCSRC_DIR}/plugin/device/gpu/kernel/cuda_impl/cuda_ops/cumsum_impl.cu
    ${CCSRC_DIR}/plugin/device/gpu/kernel/cuda_impl/cuda_ops/batchtospace_impl.cu
    ${CCSRC_DIR}/plugin/device/gpu/kernel/cuda_impl/cuda_ops/spacetobatch_impl.cu
    ${CCSRC_DIR}/plugin/device/gpu/kernel/cuda_impl/cuda_ops/depthtospace_impl.cu
    ${CCSRC_DIR}/plugin/device/gpu/kernel/cuda_impl/cuda_ops/select_impl.cu
    ${CCSRC_DIR}/plugin/device/gpu/kernel/cuda_impl/cuda_ops/maxpool_with_argmax_impl.cu
    ${CCSRC_DIR}/plugin/device/gpu/kernel/cuda_impl/cuda_ops/roi_align_impl.cu
    ${CCSRC_DIR}/plugin/device/gpu/kernel/cuda_impl/cuda_ops/nms_with_mask_impl.cu
    ${CCSRC_DIR}/plugin/device/gpu/kernel/cuda_impl/cuda_ops/boundingbox_decode_impl.cu
    ${CCSRC_DIR}/plugin/device/gpu/kernel/cuda_impl/cuda_ops/where_impl.cu
    ${CCSRC_DIR}/plugin/device/gpu/kernel/cuda_impl/cuda_ops/one_hot_impl.cu
    ${CCSRC_DIR}/plugin/device/gpu/kernel/cuda_impl/cuda_ops/tensor_scatter_arithmetic.cu
    )

set_source_files_properties(${CUDA_KERNEL_SRC} PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGES} -fPIC")
SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-arch=sm_53)
cuda_add_library(cuda_kernel_mid STATIC ${CUDA_KERNEL_SRC})
