file(GLOB_RECURSE DEVICE_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "common/*.cc"
    "kernel_info.cc" "executor/dynamic_kernel.cc" "executor/executor_callback.cc" "kernel_runtime.cc"
    "memory_manager.cc" "kernel_runtime_manager.cc" "convert_tensor_utils.cc"
    "bucket.cc" "launch_kernel.cc" "launch_mul.cc"
)

if("${ENABLE_HIDDEN}" STREQUAL "OFF")
    string(REPLACE " -Werror " " " CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    string(REPLACE " -fvisibility=hidden" " -fvisibility=default" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endif()

if(ENABLE_GPU)
    list(APPEND DEVICE_SRC_LIST "gpu/distribution/collective_init.cc")
else()
    list(APPEND DEVICE_SRC_LIST "gpu/distribution/collective_fake_init.cc")
endif()

if(ENABLE_D)
    file(GLOB_RECURSE D_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "ascend/*.cc" "kernel_adjust.cc"
        "../../minddata/dataset/engine/tdt/tdt_handle.cc")
endif()

if(ENABLE_CPU)
    file(GLOB_RECURSE CPU_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "cpu/*.cc")
    list(REMOVE_ITEM CPU_SRC_LIST "cpu/mpi/mpi_adapter.cc" "cpu/mpi/mpi_export.cc")
endif()

if(ENABLE_MPI)
    if(ENABLE_CPU)
        file(GLOB_RECURSE MPI_SRC_LIST "cpu/mpi/mpi_adapter.cc" "cpu/mpi/mpi_export.cc")
        set_property(SOURCE ${MPI_SRC_LIST}
            PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_DEVICE)
        add_library(mpi_adapter SHARED ${MPI_SRC_LIST})
        target_link_libraries(mpi_adapter PRIVATE mindspore::ompi)
    endif()

    if(ENABLE_GPU)
        set_property(SOURCE "gpu/mpi/mpi_initializer.cc"
            PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_DEVICE)
        pybind11_add_module(_ms_mpi "gpu/mpi/mpi_initializer.cc")
        target_link_libraries(_ms_mpi PRIVATE mindspore::pybind11_module mindspore::ompi)
    endif()
endif()

# gpu
if(ENABLE_GPU)
    file(GLOB_RECURSE CUDA_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "gpu/*.cc" "gpu/*.cu")

    set(GPU_QUEUE_SRCS "gpu/blocking_queue.cc" "gpu/gpu_buffer_mgr.cc")
    set(GPU_COLLECTIVE_SRCS "gpu/distribution/collective_wrapper.cc"
                            "gpu/distribution/mpi_wrapper.cc"
                            "gpu/distribution/nccl_wrapper.cc")

    # gpu_queue
    list(REMOVE_ITEM CUDA_SRC_LIST ${GPU_QUEUE_SRCS})
    set_property(SOURCE ${GPU_QUEUE_SRCS} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_DEVICE)
    add_library(gpu_queue SHARED ${GPU_QUEUE_SRCS})
    target_link_libraries(gpu_queue ${CMAKE_THREAD_LIBS_INIT} ${CUDA_PATH}/lib64/libcudart.so)

    list(REMOVE_ITEM CUDA_SRC_LIST "gpu/mpi/mpi_initializer.cc" ${GPU_COLLECTIVE_SRCS})

    if(ENABLE_MPI)
        include(ExternalProject)
        # gpu_collective
        set_property(SOURCE ${GPU_COLLECTIVE_SRCS}
            PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_DEVICE)
        add_library(gpu_collective SHARED ${GPU_COLLECTIVE_SRCS})
        target_link_libraries(gpu_collective PRIVATE mindspore::ompi mindspore::nccl)
        target_link_libraries(_ms_mpi PRIVATE gpu_collective)
    endif()

    # add_library(_mindspore_device_cuda_obj OBJECT ${CUDA_SRC_LIST})
endif()

list(REMOVE_ITEM D_SRC_LIST "ascend/profiling/profiling_callback_register.cc")
set_property(SOURCE ${DEVICE_SRC_LIST} ${D_SRC_LIST} ${CPU_SRC_LIST}
    PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_DEVICE)
add_library(_mindspore_runtime_device_obj OBJECT ${DEVICE_SRC_LIST} ${D_SRC_LIST} ${CPU_SRC_LIST})
