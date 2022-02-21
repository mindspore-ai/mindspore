if("${ENABLE_HIDDEN}" STREQUAL "OFF")
    string(REPLACE " -Werror " " " CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    string(REPLACE " -fvisibility=hidden" " -fvisibility=default" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endif()

if(ENABLE_GPU)
    list(APPEND DEVICE_SRC_LIST "distribution/collective_init.cc")
    list(APPEND DEVICE_SRC_LIST "gpu_comm_manager.cc")
else()
    list(APPEND DEVICE_SRC_LIST "distribution/collective_fake_init.cc")
endif()

if(ENABLE_GPU)
    if(ENABLE_MPI)
        set_property(SOURCE "mpi/mpi_initializer.cc"
                PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_DEVICE)
        pybind11_add_module(_ms_mpi NO_EXTRAS "mpi/mpi_initializer.cc")
        target_link_libraries(_ms_mpi PRIVATE mindspore::pybind11_module mindspore::ompi)
    endif()

    file(GLOB_RECURSE CUDA_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc" "*.cu")

    set(GPU_QUEUE_SRCS "blocking_queue.cc" "gpu_buffer_mgr.cc")
    set(GPU_COLLECTIVE_SRCS "distribution/collective_wrapper.cc"
                            "distribution/mpi_wrapper.cc"
                            "distribution/nccl_wrapper.cc")

    # gpu_queue
    list(REMOVE_ITEM CUDA_SRC_LIST ${GPU_QUEUE_SRCS})
    set_property(SOURCE ${GPU_QUEUE_SRCS} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_DEVICE)
    add_library(gpu_queue SHARED ${GPU_QUEUE_SRCS})
    target_link_libraries(gpu_queue ${CMAKE_THREAD_LIBS_INIT} ${CUDA_PATH}/lib64/libcudart.so)

    list(REMOVE_ITEM CUDA_SRC_LIST "mpi/mpi_initializer.cc" ${GPU_COLLECTIVE_SRCS})

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

set_property(SOURCE ${DEVICE_SRC_LIST} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_DEVICE)
add_library(_mindspore_plugin_device_gpu_hal_device_obj OBJECT ${DEVICE_SRC_LIST})

