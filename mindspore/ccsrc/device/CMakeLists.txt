file(GLOB_RECURSE _DEVICE_ALL_SRC_FILES *.cc)
add_library(_mindspore_device_obj OBJECT ${_DEVICE_ALL_SRC_FILES})

if(ENABLE_CPU)
    target_compile_definitions(_mindspore_device_obj PRIVATE CPUSESSION)
    file(GLOB_RECURSE _CPU_SRC_LIST  cpu/*.cc)
    add_library(_c_expression_cpu_device_obj  OBJECT ${_CPU_SRC_LIST})
endif()

if(ENABLE_GPU)
    file(GLOB_RECURSE _GPU_SRC_LIST  RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
            "gpu/*.cc"
            "gpu/*.cu"
            )
    list(REMOVE_ITEM _GPU_SRC_LIST "gpu/blocking_queue.cc"
                                   "gpu/gpu_buffer_mgr.cc"
                                   "gpu/mpi/mpi_initializer.cc"
                                   "gpu/distribution/collective_wrapper.cc"
                                   "gpu/distribution/mpi_wrapper.cc"
                                   "gpu/distribution/nccl_wrapper.cc")
    add_library(_cuda_gpu_device_obj  OBJECT ${_GPU_SRC_LIST})
endif()

if(ENABLE_D)
    file(GLOB_RECURSE _D_SRC_LIST  RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
            "ascend/*.cc"
            "ascend/profiling/*.cc"
            "ascend/tasksink/*.cc"
            "kernel_adjust.cc"
            "ascend/tasksink/taskinfo/*.cc"
            )
    target_sources(_mindspore_device_obj  PRIVATE ${_D_SRC_LIST})
endif()

