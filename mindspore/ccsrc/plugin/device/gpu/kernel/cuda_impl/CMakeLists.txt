file(GLOB_RECURSE CUDA_OPS_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cu")

if(CMAKE_SYSTEM_NAME MATCHES "Darwin")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-delete-non-abstract-non-virtual-dtor -Wno-overloaded-virtual")
endif()

if(${CUDA_VERSION} VERSION_LESS 11.0)
    if(NOT MSVC)
    string(REPLACE "-std=c++17" "-std=c++11" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    endif()
else()
    if(NOT MSVC)
    string(REPLACE "-std=c++17" "-std=c++14" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    endif()
endif()

set_property(SOURCE ${CUDA_OPS_SRC_LIST} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_KERNEL)
if(MSVC)
    add_compile_definitions(BUILDING_CUDA_OPS_DLL)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -DBUILDING_CUDA_OPS_DLL -std=c++17")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -D_DISABLE_EXTENDED_ALIGNED_STORAGE")
endif()
if(ENABLE_GPU)
    add_library(cuda_common_obj OBJECT cuda_ops/cuda_common.cc  cuda_ops/cuda_device_info.cc)
    if(NOT MSVC)
    target_compile_options(cuda_common_obj PRIVATE "-std=c++17")
    else()
    target_compile_options(cuda_common_obj PRIVATE "/std:c++17")
    endif()
    cuda_add_library(cuda_ops SHARED ${CUDA_OPS_SRC_LIST} $<TARGET_OBJECTS:cuda_common_obj>)
    message("add gpu lib to cuda_ops")
    if(WIN32)
    target_link_libraries(cuda_ops mindspore_core
            ${CUBLAS_LIBRARY_PATH}
            ${CUDA_PATH}/lib/x64/curand.lib
            ${CUDNN_LIBRARY_PATH}
            ${CUDA_PATH}/lib/x64/cudart.lib
            ${CUDA_PATH}/lib/x64/cuda.lib
            ${CUDA_PATH}/lib/x64/cusolver.lib
            ${CUDA_PATH}/lib/x64/cufft.lib)
    else()
    target_link_libraries(cuda_ops mindspore_core
            ${CUBLAS_LIBRARY_PATH}
            ${CUDA_PATH}/lib64/libcurand.so
            ${CUDNN_LIBRARY_PATH}
            ${CUDA_PATH}/lib64/libcudart.so
            ${CUDA_PATH}/lib64/stubs/libcuda.so
            ${CUDA_PATH}/lib64/libcusolver.so
            ${CUDA_PATH}/lib64/libcufft.so)
    endif()
endif()