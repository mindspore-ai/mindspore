file(GLOB_RECURSE DEVICE_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "common/*.cc"
    "kernel_info.cc" "executor/dynamic_kernel.cc" "executor/executor_callback.cc" "kernel_runtime.cc"
    "memory_manager.cc" "kernel_runtime_manager.cc" "convert_tensor_utils.cc" "memory_scheduler.cc"
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
    file(GLOB_RECURSE D_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "ascend/*.cc" "kernel_adjust.cc")
    list(REMOVE_ITEM D_SRC_LIST "ascend/distribute/mpi_collective_group.cc"
            "ascend/distribute/collective_group_wrapper.cc" "ascend/distribute/mpi_pycc.cc")
endif()
if(ENABLE_TDTQUE)
    file(GLOB_RECURSE TDT_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
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
        target_link_libraries(mpi_adapter PRIVATE mindspore::ompi mindspore::pybind11_module -ldl ${SECUREC_LIBRARY})
    endif()

    if(ENABLE_GPU)
        set_property(SOURCE "gpu/mpi/mpi_initializer.cc"
            PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_DEVICE)
        pybind11_add_module(_ms_mpi NO_EXTRAS "gpu/mpi/mpi_initializer.cc")
        target_link_libraries(_ms_mpi PRIVATE mindspore::pybind11_module mindspore::ompi)
    endif()

    if(ENABLE_D)
        set_property(SOURCE "ascend/distribute/mpi_pycc.cc"
                PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_DEVICE)
        pybind11_add_module(_ascend_mpi "ascend/distribute/mpi_pycc.cc")
        target_link_libraries(_ascend_mpi PRIVATE mindspore::pybind11_module mindspore::ompi)
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

if(ENABLE_SECURITY)
    list(REMOVE_ITEM D_SRC_LIST "ascend/profiling/profiling_callback_register.cc")
    list(REMOVE_ITEM D_SRC_LIST "ascend/profiling/profiling_manager.cc")
    list(REMOVE_ITEM D_SRC_LIST "ascend/profiling/profiling_utils.cc")
    list(REMOVE_ITEM D_SRC_LIST "ascend/profiling/reporter/desc_reporter.cc")
    list(REMOVE_ITEM D_SRC_LIST "ascend/profiling/reporter/graph_desc_reporter.cc")
    list(REMOVE_ITEM D_SRC_LIST "ascend/profiling/reporter/op_name_task_stream_reporter.cc")
    list(REMOVE_ITEM D_SRC_LIST "ascend/profiling/reporter/point_reporter.cc")
    list(REMOVE_ITEM D_SRC_LIST "ascend/profiling/reporter/profiling_desc.cc")
    list(REMOVE_ITEM D_SRC_LIST "ascend/profiling/reporter/task_desc_reporter.cc")
endif()

set_property(SOURCE ${DEVICE_SRC_LIST} ${D_SRC_LIST} ${CPU_SRC_LIST}
    PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_DEVICE)
add_library(_mindspore_runtime_device_obj OBJECT ${DEVICE_SRC_LIST} ${D_SRC_LIST} ${CPU_SRC_LIST} ${TDT_SRC_LIST})
if(ENABLE_D)
    file(GLOB_RECURSE GE_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "ascend/ge_runtime/*.cc")
    set_property(SOURCE ${GE_SRC_LIST} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_GE)
    target_include_directories(_mindspore_runtime_device_obj PRIVATE ${CMAKE_BINARY_DIR}/proto/ge)
    add_dependencies(_mindspore_runtime_device_obj graph)
    if(ENABLE_MPI)
        set(ASCEND_PATH /usr/local/Ascend)
        set(ASCEND_TOOLKIT_RUNTIME_PATH ${ASCEND_PATH}/ascend-toolkit/latest/fwkacllib/lib64)
        set(ASCEND_RUNTIME_PATH ${ASCEND_PATH}/fwkacllib/lib64)
        find_library(HCCL hccl ${ASCEND_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH})
        file(GLOB_RECURSE ASCEND_COLLECTIVE_LIST "ascend/distribute/mpi_collective_group.cc"
                "ascend/distribute/collective_group_wrapper.cc")
        set_property(SOURCE ${ASCEND_COLLECTIVE_LIST}
                PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_DEVICE)
        add_library(ascend_collective SHARED ${ASCEND_COLLECTIVE_LIST})
        target_link_libraries(ascend_collective PRIVATE ${HCCL} mindspore::ompi)
        target_link_libraries(_ascend_mpi PRIVATE ascend_collective)
    endif()
endif()
