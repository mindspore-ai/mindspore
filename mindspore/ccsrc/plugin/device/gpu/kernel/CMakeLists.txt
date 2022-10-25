if(CMAKE_SYSTEM_NAME MATCHES "Darwin")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-delete-non-abstract-non-virtual-dtor -Wno-overloaded-virtual")
endif()

if(ENABLE_GPU)
    file(GLOB_RECURSE GPU_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")
    file(GLOB_RECURSE _AKG_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "akg/*.cc")
    list(REMOVE_ITEM GPU_SRC_LIST ${_AKG_SRC_LIST})
    list(REMOVE_ITEM GPU_SRC_LIST "nccl/nccl_collective_gpu_kernel.cc")
    list(REMOVE_ITEM GPU_SRC_LIST "nccl/nccl_p2p_gpu_kernel.cc")
    list(REMOVE_ITEM GPU_SRC_LIST "nccl/nccl_send_gpu_kernel.cc")
    list(REMOVE_ITEM GPU_SRC_LIST "nccl/nccl_recv_gpu_kernel.cc")
    list(REMOVE_ITEM GPU_SRC_LIST "trt/trt_kernel.cc")

    if(WIN32)
        list(REMOVE_ITEM GPU_SRC_LIST "nccl/nccl_gpu_kernel.cc")
        list(REMOVE_ITEM GPU_SRC_LIST "rl/mux_base_gpu_kernel.cc")
        list(REMOVE_ITEM GPU_SRC_LIST "rl/mux_send_gpu_kernel.cc")
        list(REMOVE_ITEM GPU_SRC_LIST "rl/mux_recv_gpu_kernel.cc")
    endif()

    if(NOT (CMAKE_SYSTEM_NAME MATCHES "Linux" AND ${CUDA_VERSION} VERSION_GREATER "11.0"))
        list(REMOVE_ITEM GPU_SRC_LIST "map_tensor/map_tensor_erase_gpu_kernel.cc")
        list(REMOVE_ITEM GPU_SRC_LIST "map_tensor/map_tensor_get_data_gpu_kernel.cc")
        list(REMOVE_ITEM GPU_SRC_LIST "map_tensor/map_tensor_get_gpu_kernel.cc")
        list(REMOVE_ITEM GPU_SRC_LIST "map_tensor/map_tensor_put_gpu_kernel.cc")
        list(REMOVE_ITEM GPU_SRC_LIST "map_tensor/map_tensor_get_grad_gpu_kernel.cc")
    endif()

    if(ENABLE_MPI)
        include(ExternalProject)
        file(GLOB_RECURSE GPU_NCCL_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "nccl/*.cc")
        list(APPEND GPU_SRC_LIST ${GPU_NCCL_LIST})
    endif()

    if(ENABLE_GPU_INFER)
        file(GLOB_RECURSE GPU_TRT_KERNEL_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "trt/*.cc")
        list(APPEND GPU_SRC_LIST ${GPU_TRT_KERNEL_LIST})
    endif()
endif()

if(ENABLE_AKG AND ${CMAKE_SYSTEM_NAME} MATCHES "Linux" AND ENABLE_GPU)
    file(GLOB_RECURSE AKG_GPU_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        "akg/*.cc"
    )
endif()

set_property(SOURCE ${GPU_SRC_LIST} ${AKG_GPU_SRC_LIST}
        PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_KERNEL)
if(ENABLE_GPU)
    add_library(_mindspore_plugin_device_gpu_kernel_obj OBJECT ${GPU_SRC_LIST} ${AKG_GPU_SRC_LIST})
endif()
