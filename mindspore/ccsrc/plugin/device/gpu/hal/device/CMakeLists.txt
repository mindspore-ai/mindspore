if("${ENABLE_HIDDEN}" STREQUAL "OFF" AND NOT MSVC)
    string(REPLACE " -Werror " " " CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    string(REPLACE " -fvisibility=hidden" " -fvisibility=default" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endif()

list(APPEND DEVICE_SRC_LIST "ps/gpu_ps_cache.cc")
if(ENABLE_GPU)
    list(APPEND DEVICE_SRC_LIST ${CMAKE_SOURCE_DIR}/mindspore/ccsrc/backend/common/mem_reuse/mem_reuse.cc)
    list(APPEND DEVICE_SRC_LIST ${CMAKE_SOURCE_DIR}/mindspore/ccsrc/backend/common/mem_reuse/mem_swap_manager.cc)
    list(APPEND DEVICE_SRC_LIST ${CMAKE_SOURCE_DIR}/mindspore/ccsrc/runtime/data_queue/data_queue.h)
    file(GLOB_RECURSE DEVICE_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")
    list(REMOVE_ITEM DEVICE_SRC_LIST
            "distribution/collective_wrapper.cc"
            "distribution/mpi_wrapper.cc"
            "distribution/nccl_wrapper.cc"
            "trt_loader.cc")
    if(NOT ${TENSORRT_HOME} STREQUAL "")
        find_path(TENSORRT_HOME_INCLUDE NvInfer.h HINTS ${TENSORRT_HOME}/include)
        if(TENSORRT_HOME_INCLUDE STREQUAL TENSORRT_HOME_INCLUDE-NOTFOUND)
            message(FATAL_ERROR "Tensor-RT dir not exist ${TENSORRT_HOME}")
        endif()
        message("Enable GPU inference. Tensor-RT include dir: ${TENSORRT_HOME_INCLUDE}")
        set(ENABLE_GPU_INFER TRUE)
        add_compile_definitions(ENABLE_GPU_INFER)
        include_directories(${TENSORRT_HOME_INCLUDE})
        list(APPEND DEVICE_SRC_LIST ${CMAKE_CURRENT_SOURCE_DIR}/trt_loader.cc)
    endif()
endif()

if(ENABLE_GPU)
    file(GLOB_RECURSE CUDA_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}  "*.cu")
    set_property(SOURCE ${CUDA_SRC_LIST} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_DEVICE)

    if(CMAKE_SYSTEM_NAME MATCHES "Linux" AND ${CUDA_VERSION} VERSION_GREATER "11.0")
        string(REPLACE "-arch=sm_53;" "" CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}")
        string(REPLACE "-gencode=arch=compute_53,code=sm_53;" "" CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}")
        list(APPEND CUDA_NVCC_FLAGS -gencode=arch=compute_70,code=sm_70)
        list(APPEND CUDA_NVCC_FLAGS -gencode=arch=compute_75,code=sm_75)
        list(APPEND CUDA_NVCC_FLAGS -gencode=arch=compute_80,code=sm_80)
        list(APPEND CUDA_NVCC_FLAGS -gencode=arch=compute_86,code=compute_86)
        list(APPEND CUDA_NVCC_FLAGS -std=c++17)
        list(REMOVE_DUPLICATES CUDA_NVCC_FLAGS)
        cuda_add_library(gpu_hash_table STATIC ${CUDA_SRC_LIST})
    endif()
endif()

set_property(SOURCE ${DEVICE_SRC_LIST} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_DEVICE)
add_library(_mindspore_plugin_device_gpu_hal_device_obj OBJECT ${DEVICE_SRC_LIST})

