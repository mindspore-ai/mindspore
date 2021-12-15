include_directories(${TENSORRT_PATH}/include)
include_directories(${CUDA_PATH}/include)
include_directories($(CCSRC_DIR)/backend/kernel_compiler/cpu)

if(DEFINED ENV{MS_ENABLE_CUDA_DISTRIBUTION})
    set(MS_ENABLE_CUDA_DISTRIBUTION $ENV{MS_ENABLE_CUDA_DISTRIBUTION})
else()
    set(MS_ENABLE_CUDA_DISTRIBUTION "off")
endif()

set(NCCL_MPI_SRC_STUB
    ${CMAKE_CURRENT_SOURCE_DIR}/distribution/distribution_collective.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/distribution/distribution_base.cc
)

# nccl mpi
if(MS_ENABLE_CUDA_DISTRIBUTION STREQUAL "on")
    message("enable cuda gpu distribution collective")
    file(GLOB NCCL_MPI_SRC LIST_DIRECTORIES false
        ${CMAKE_CURRENT_SOURCE_DIR}/distribution/*.cc
        ${CCSRC_DIR}/runtime/device/gpu/distribution/collective_wrapper.cc
        ${CCSRC_DIR}/runtime/device/gpu/distribution/mpi_wrapper.cc
        ${CCSRC_DIR}/runtime/device/gpu/distribution/nccl_wrapper.cc
    )
    list(REMOVE_ITEM NCCL_MPI_SRC ${NCCL_MPI_SRC_STUB})

    add_compile_definitions(LITE_CUDA_DISTRIBUTION)
    include(${TOP_DIR}/cmake/external_libs/ompi.cmake)
    include(${TOP_DIR}/cmake/external_libs/nccl.cmake)

    add_library(gpu_distribution_collective OBJECT ${NCCL_MPI_SRC})
    add_library(mindspore::nccl ALIAS nccl::nccl)
    add_library(mindspore::ompi ALIAS ompi::mpi)
    target_link_libraries(gpu_distribution_collective PRIVATE mindspore::ompi mindspore::nccl)
else()
    add_library(gpu_distribution_collective OBJECT ${NCCL_MPI_SRC_STUB})
endif()
add_dependencies(gpu_distribution_collective fbs_src)

# lstm only compile when tensorrt version is later than 7
file(GLOB TENSORRT_LIB_FILE false ${TENSORRT_LIB_PATH}/libnvinfer.so*)
set(HIGH_TRT_VERSION "false")
FOREACH(TENSORRT_LIB ${TENSORRT_LIB_FILE})
    STRING(REGEX MATCHALL "(.*libnvinfer\.so\.)([1-9])\.([1-9]\.*)+" VERSION_NAME ${TENSORRT_LIB})
    IF(VERSION_NAME STREQUAL "")
        CONTINUE()
    ENDIF()
    STRING(REGEX REPLACE ".*libnvinfer\.so\." "" VERSION_NAME ${VERSION_NAME})
    STRING(REGEX MATCH "[1-9]" MAIN_VERSION ${VERSION_NAME})
    MESSAGE("compile with TensorRT version: ${MAIN_VERSION}")
    IF(MAIN_VERSION STREQUAL "7" OR "8" OR "9")
        set(HIGH_TRT_VERSION "true")
    ENDIF()
ENDFOREACH()
IF(HIGH_TRT_VERSION STREQUAL "false")
    MESSAGE("TensorRT version is too low to get new features, please upgrade TesorRT version to 7 or higher.")
ELSE()
    add_compile_definitions(HIGH_TRT_VERSION)
ENDIF()

file(GLOB TENSORRT_RUNTIME_SRC LIST_DIRECTORIES false
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/op/*.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda_impl/*.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../delegate_utils.cc
)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../parameter_cache)

set(TENSORRT_RUNTIME_SRC
        ${TENSORRT_RUNTIME_SRC}
        ${CMAKE_CURRENT_SOURCE_DIR}/../parameter_cache/embedding_cache_manager.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../parameter_cache/load_host_cache_model.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../parameter_cache/lfu_cache.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../parameter_cache/embedding_cache.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../parameter_cache/gpu/gpu_cache_mem.cc
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

