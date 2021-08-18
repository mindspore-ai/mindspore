include_directories(${TENSORRT_PATH}/include)
include_directories(${CUDA_PATH}/include)
file(GLOB_RECURSE TENSORRT_RUNTIME_SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/op/*.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../delegate_utils.cc
        )
add_library(libcudart SHARED IMPORTED)
set_target_properties(libcudart PROPERTIES IMPORTED_LOCATION ${CUDA_LIB_PATH}/libcudart.so)

add_library(libcudnn SHARED IMPORTED)
set_target_properties(libcudnn PROPERTIES IMPORTED_LOCATION ${CUDA_LIB_PATH}/libcudnn.so)

add_library(libnvrtc SHARED IMPORTED)
set_target_properties(libnvrtc PROPERTIES IMPORTED_LOCATION ${CUDA_LIB_PATH}/libnvrtc.so)

add_library(libcublas SHARED IMPORTED)
set_target_properties(libcublas PROPERTIES IMPORTED_LOCATION ${CUDA_LIB_PATH}/libcublas.so)

add_library(libcublasLt SHARED IMPORTED)
set_target_properties(libcublasLt PROPERTIES IMPORTED_LOCATION ${CUDA_LIB_PATH}/libcublasLt.so)

add_library(libnvinfer SHARED IMPORTED)
set_target_properties(libnvinfer PROPERTIES IMPORTED_LOCATION ${TENSORRT_LIB_PATH}/libnvinfer.so)

add_library(tensorrt_kernel_mid OBJECT ${TENSORRT_RUNTIME_SRC})
add_dependencies(tensorrt_kernel_mid fbs_src)
target_link_libraries(
        tensorrt_kernel_mid
        libcudart
        libcudnn
        libnvrtc
        libcublas
        libcublasLt
        libnvinfer
)
