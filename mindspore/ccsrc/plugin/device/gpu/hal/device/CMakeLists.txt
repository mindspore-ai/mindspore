if("${ENABLE_HIDDEN}" STREQUAL "OFF" AND NOT MSVC)
    string(REPLACE " -Werror " " " CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    string(REPLACE " -fvisibility=hidden" " -fvisibility=default" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endif()

list(APPEND DEVICE_SRC_LIST "ps/gpu_ps_cache.cc")
if(ENABLE_GPU)
    list(APPEND DEVICE_SRC_LIST ${CMAKE_SOURCE_DIR}/mindspore/ccsrc/common/mem_reuse/mem_reuse.cc)
    list(APPEND DEVICE_SRC_LIST ${CMAKE_SOURCE_DIR}/mindspore/ccsrc/common/mem_reuse/mem_swap_manager.cc)
    list(APPEND DEVICE_SRC_LIST ${CMAKE_SOURCE_DIR}/mindspore/ccsrc/runtime/data_queue/data_queue.h)
    file(GLOB_RECURSE DEVICE_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")
    list(REMOVE_ITEM DEVICE_SRC_LIST
            "mpi/mpi_initializer.cc"
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
    if(ENABLE_MPI)
        set_property(SOURCE "mpi/mpi_initializer.cc"
                PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_DEVICE)
        pybind11_add_module(_ms_mpi NO_EXTRAS "mpi/mpi_initializer.cc")
        target_link_libraries(_ms_mpi PRIVATE mindspore::pybind11_module mindspore::ompi)
    endif()

    file(GLOB_RECURSE CUDA_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}  "*.cu")
    set_property(SOURCE ${CUDA_SRC_LIST} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_DEVICE)

    if(CMAKE_SYSTEM_NAME MATCHES "Linux" AND ${CUDA_VERSION} VERSION_GREATER "11.0")
        string(REPLACE "-arch=sm_53;" "" CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}")
        list(APPEND CUDA_NVCC_FLAGS -std=c++17)
        cuda_add_library(gpu_hash_table STATIC ${CUDA_SRC_LIST})
    endif()

    set(GPU_COLLECTIVE_SRCS "distribution/collective_wrapper.cc"
                            "distribution/mpi_wrapper.cc"
                            "distribution/nccl_wrapper.cc")


    if(ENABLE_MPI)
        include(ExternalProject)
        # gpu_collective
        set_property(SOURCE ${GPU_COLLECTIVE_SRCS}
            PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_DEVICE)
        add_library(gpu_collective SHARED ${GPU_COLLECTIVE_SRCS})
        target_link_libraries(gpu_collective PRIVATE mindspore::ompi mindspore::nccl)
        target_link_libraries(_ms_mpi PRIVATE gpu_collective)
    endif()
endif()

set_property(SOURCE ${DEVICE_SRC_LIST} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_DEVICE)
add_library(_mindspore_plugin_device_gpu_hal_device_obj OBJECT ${DEVICE_SRC_LIST})

