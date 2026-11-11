include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CUDNN_INCLUDE_PATH} ${CUDA_PATH} ${CUDA_INCLUDE_DIRS} ${CUPTI_INCLUDE_DIRS})

if(CMAKE_SYSTEM_NAME MATCHES "Darwin")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-delete-non-abstract-non-virtual-dtor -Wno-overloaded-virtual")
endif()

if(ENABLE_GPU)
    file(GLOB_RECURSE GPU_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
            "*.cc"
    )

    file(GLOB_RECURSE PYBOOST_GPU_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
            "pyboost/*.cc"
            "pyboost/auto_generate/*.cc"
            "pyboost/ops/*.cc"
            )
    list(APPEND GPU_SRC_LIST "${PYBOOST_GPU_SRC_LIST}")

    file(GLOB_RECURSE _AKG_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        "akg/*.cc"
        "dynamic_akg/*.cc"
    )
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
        "dynamic_akg/*.cc"
    )
endif()

set_property(SOURCE ${GPU_SRC_LIST} ${AKG_GPU_SRC_LIST}
        PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_KERNEL)

if(ENABLE_GPU)
    add_library(mindspore_ops_gpu STATIC ${GPU_SRC_LIST} ${AKG_GPU_SRC_LIST})
    if(CMAKE_SYSTEM_NAME MATCHES "Windows")
        target_compile_definitions(mindspore_ops_gpu PRIVATE BUILDING_GPU_DLL)
    endif()

    add_subdirectory(cuda_impl)
    if(WIN32)
        target_link_libraries(mindspore_ops_gpu PRIVATE cuda_ops
                ${CUBLAS_LIBRARY_PATH}
                ${CUDA_PATH}/lib/x64/curand.lib
                ${CUDNN_LIBRARY_PATH}
                ${CUDA_PATH}/lib/x64/cudart.lib
                ${CUDA_PATH}/lib/x64/cuda.lib
                ${CUDA_PATH}/lib/x64/cusolver.lib
                ${CUDA_PATH}/lib/x64/cufft.lib
                ${CUDA_PATH}/lib/x64/cusparse.lib
                ${CUDA_PATH}/lib/x64/nvrtc.lib
                )
    else()
        target_link_libraries(mindspore_ops_gpu PRIVATE cuda_ops
                ${CUBLAS_LIBRARY_PATH}
                ${CUDA_PATH}/lib64/libcurand.so
                ${CUDNN_LIBRARY_PATH}
                ${CUDA_PATH}/lib64/libcudart.so
                ${CUDA_PATH}/lib64/stubs/libcuda.so
                ${CUDA_PATH}/lib64/libcusolver.so
                ${CUDA_PATH}/lib64/libcufft.so
                ${CUDA_PATH}/lib64/libcusparse.so
                ${CUDA_PATH}/lib64/libnvrtc.so
                )
    endif()
endif()
