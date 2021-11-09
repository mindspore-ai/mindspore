include_directories(${TENSORRT_PATH}/include)
include_directories(${CUDA_PATH}/include)
include_directories($(CCSRC_DIR)/backend/kernel_compiler/cpu)
file(GLOB_RECURSE TENSORRT_RUNTIME_SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/op/*.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../delegate_utils.cc
        )

link_libraries(${CUDA_LIB_PATH}/libcudnn.so)
link_libraries(${CUDA_LIB_PATH}/libnvrtc.so)
link_libraries(${CUDA_LIB_PATH}/libcublas.so)
link_libraries(${CUDA_LIB_PATH}/libcublasLt.so)

add_library(libcudart SHARED IMPORTED)
set_target_properties(libcudart PROPERTIES IMPORTED_LOCATION ${CUDA_LIB_PATH}/libcudart.so)

add_library(libnvinfer SHARED IMPORTED)
set_target_properties(libnvinfer PROPERTIES IMPORTED_LOCATION ${TENSORRT_LIB_PATH}/libnvinfer.so)

add_library(tensorrt_kernel_mid OBJECT ${TENSORRT_RUNTIME_SRC})
add_dependencies(tensorrt_kernel_mid fbs_src)
target_link_libraries(
        tensorrt_kernel_mid
        libcudart
        libnvinfer
)

# cuda
find_package(CUDA)
file(GLOB_RECURSE CUDA_KERNEL_SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/cuda_impl/*.cu
)
set_source_files_properties(${CUDA_KERNEL_SRC} PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGES} -std=c++14 -fPIC")
SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-std=c++14;)
cuda_add_library(cuda_kernel_mid STATIC ${CUDA_KERNEL_SRC})
