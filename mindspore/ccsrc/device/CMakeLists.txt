file(GLOB_RECURSE DEVICE_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "common/*.cc"
    "kernel_info.cc" "kernel_runtime.cc" "memory_manager.cc" "kernel_runtime_manager.cc" "convert_tensor_utils.cc"
)

if (ENABLE_GPU)
    list(APPEND DEVICE_SRC_LIST "gpu/distribution/collective_init.cc")
else ()
    list(APPEND DEVICE_SRC_LIST "gpu/distribution/collective_fake_init.cc")
endif ()

if (ENABLE_D)
    file(GLOB_RECURSE D_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "ascend/*.cc" "kernel_adjust.cc")
endif ()

if (ENABLE_CPU)
    file(GLOB_RECURSE CPU_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "cpu/*.cc")
endif ()

# gpu
if (ENABLE_GPU)
    file(GLOB_RECURSE CUDA_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "gpu/*.cc" "gpu/*.cu")

    # gpu_queue
    list(REMOVE_ITEM CUDA_SRC_LIST "gpu/blocking_queue.cc" "gpu/gpu_buffer_mgr.cc")
    add_library(gpu_queue SHARED "gpu/blocking_queue.cc" "gpu/gpu_buffer_mgr.cc")
    target_link_libraries(gpu_queue ${CMAKE_THREAD_LIBS_INIT} ${CUDA_PATH}/lib64/libcudart.so)

    list(REMOVE_ITEM CUDA_SRC_LIST "gpu/mpi/mpi_initializer.cc"
                                   "gpu/distribution/collective_wrapper.cc"
                                   "gpu/distribution/mpi_wrapper.cc"
                                   "gpu/distribution/nccl_wrapper.cc"
    )

    if (ENABLE_MPI)
        include(ExternalProject)
        # gpu_collective
        add_library(gpu_collective SHARED "gpu/distribution/collective_wrapper.cc"
                                          "gpu/distribution/mpi_wrapper.cc"
                                          "gpu/distribution/nccl_wrapper.cc"
        )
        # _ms_mpi
        pybind11_add_module(_ms_mpi "gpu/mpi/mpi_initializer.cc")
        target_link_libraries(_ms_mpi PRIVATE mindspore::pybind11_module mindspore::ompi)
        target_link_libraries(gpu_collective PRIVATE mindspore::ompi mindspore::nccl)
    endif ()

    # add_library(_mindspore_device_cuda_obj OBJECT ${CUDA_SRC_LIST})
endif ()

set_property(SOURCE ${DEVICE_SRC_LIST} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_DEVICE)
add_library(_mindspore_device_obj OBJECT ${DEVICE_SRC_LIST} ${D_SRC_LIST} ${CPU_SRC_LIST})
