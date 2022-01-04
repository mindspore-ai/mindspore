file(GLOB_RECURSE KERNEL_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
    "kernel_build_info.cc"
    "kernel.cc"
    "kash/*.cc"
    "common_utils.cc"
    "oplib/*.cc"
    "environ_manager.cc"
)

if(CMAKE_SYSTEM_NAME MATCHES "Darwin")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-delete-non-abstract-non-virtual-dtor -Wno-overloaded-virtual")
endif()

if(ENABLE_D)
    file(GLOB_RECURSE D_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        "ascend_kernel_mod.cc"
        "kernel_query.cc"
        "tbe/*.cc"
        "host/*.cc"
        "aicpu/*.cc"
        "rts/*.cc"
        "hccl/*.cc"
    )
    file(GLOB_RECURSE AICPU_OPS_SRC RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        "aicpu/aicpu_ops/*.cc"
    )
    list(REMOVE_ITEM D_SRC_LIST ${AICPU_OPS_SRC})
    add_compile_definitions(ENABLE_D)
endif()

if(ENABLE_CPU)
    file(GLOB CPU_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        "cpu/*.cc"
        "cpu/eigen/*.cc"
        "cpu/mkldnn/*.cc"
        "cpu/fl/*.cc"
        "cpu/ps/*.cc"
        "cpu/pyfunc/*.cc"
        "cpu/rl/*.cc"
        "cpu/custom/*.cc"
        "cpu/environ/*.cc"
    )

    if(NOT ENABLE_MPI)
        list(REMOVE_ITEM CPU_SRC_LIST "cpu/allgather_cpu_kernel.cc")
        list(REMOVE_ITEM CPU_SRC_LIST "cpu/reduce_scatter_cpu_kernel.cc")
        list(REMOVE_ITEM CPU_SRC_LIST "cpu/embedding_look_up_comm_grad_cpu_kernel.cc")
    endif()

    if(PLATFORM_ARM64)
        add_compile_definitions(ENABLE_ARM)
    endif()
endif()

if(NOT ENABLE_CPU OR WIN32)
    list(REMOVE_ITEM CPU_SRC_LIST "cpu/ps/apply_momentum_ps_kernel.cc")
    list(REMOVE_ITEM CPU_SRC_LIST "cpu/ps/embedding_look_up_proxy_kernel.cc")
    list(REMOVE_ITEM CPU_SRC_LIST "cpu/ps/embedding_look_up_ps_kernel.cc")
    list(REMOVE_ITEM CPU_SRC_LIST "cpu/ps/pserver_kernel.cc")
    list(REMOVE_ITEM CPU_SRC_LIST "cpu/ps/pull_kernel.cc")
    list(REMOVE_ITEM CPU_SRC_LIST "cpu/ps/push_kernel.cc")
    list(REMOVE_ITEM CPU_SRC_LIST "cpu/ps/sparse_apply_adam_ps_kernel.cc")
    list(REMOVE_ITEM CPU_SRC_LIST "cpu/ps/sparse_apply_ftrl_ps_kernel.cc")
    list(REMOVE_ITEM CPU_SRC_LIST "cpu/ps/sparse_apply_lazy_adam_ps_kernel.cc")
    list(REMOVE_ITEM CPU_SRC_LIST "cpu/fl/fused_pull_weight_kernel.cc")
    list(REMOVE_ITEM CPU_SRC_LIST "cpu/fl/fused_push_weight_kernel.cc")
    list(REMOVE_ITEM CPU_SRC_LIST "cpu/fl/get_model_kernel.cc")
    list(REMOVE_ITEM CPU_SRC_LIST "cpu/fl/start_fl_job_kernel.cc")
    list(REMOVE_ITEM CPU_SRC_LIST "cpu/fl/update_model_kernel.cc")
    list(REMOVE_ITEM CPU_SRC_LIST "cpu/fl/push_metrics_kernel.cc")
    list(REMOVE_ITEM CPU_SRC_LIST "cpu/fl/get_keys_kernel.cc")
    list(REMOVE_ITEM CPU_SRC_LIST "cpu/fl/exchange_keys_kernel.cc")
endif()

if(ENABLE_SECURITY)
    list(REMOVE_ITEM D_SRC_LIST "rts/profiling_kernel_mod.cc")
endif()

if(ENABLE_GPU)
    file(GLOB_RECURSE CUDA_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        "gpu/*.cu"
    )

    file(GLOB_RECURSE GPU_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "gpu/*.cc")
    list(REMOVE_ITEM GPU_SRC_LIST "gpu/nccl/nccl_collective_gpu_kernel.cc")
    list(REMOVE_ITEM GPU_SRC_LIST "gpu/nccl/nccl_p2p_gpu_kernel.cc")
    list(REMOVE_ITEM GPU_SRC_LIST "gpu/nccl/nccl_send_gpu_kernel.cc")
    list(REMOVE_ITEM GPU_SRC_LIST "gpu/nccl/nccl_recv_gpu_kernel.cc")
    list(REMOVE_ITEM GPU_SRC_LIST "gpu/trt/trt_kernel.cc")

    if(ENABLE_MPI)
        include(ExternalProject)
        file(GLOB_RECURSE GPU_NCCL_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "gpu/nccl/*.cc")
        list(APPEND GPU_SRC_LIST ${GPU_NCCL_LIST})
    endif()

    if(ENABLE_GPU_INFER)
        file(GLOB_RECURSE GPU_TRT_KERNEL_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "gpu/trt/*.cc")
        list(APPEND GPU_SRC_LIST ${GPU_TRT_KERNEL_LIST})
    endif()

    # add_library(_mindspore_kernel_cuda_obj OBJECT ${CUDA_SRC_LIST})
endif()

if(ENABLE_AKG AND ${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    file(GLOB_RECURSE AKG_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        "akg/akg_kernel_build.cc"
        "akg/akg_kernel_json_generator.cc"
        "akg/akg_kernel_json_decoder.cc"
    )
    if(ENABLE_GPU)
        file(GLOB_RECURSE AKG_GPU_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
            "akg/gpu/*.cc"
        )
        list(APPEND AKG_SRC_LIST ${AKG_GPU_SRC_LIST})
    endif()
    if(ENABLE_D)
        file(GLOB_RECURSE AKG_D_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
            "akg/ascend/*.cc"
            "akg/akg_kernel_metadata.cc"
        )
        list(APPEND AKG_SRC_LIST ${AKG_D_SRC_LIST})
    endif()
    if(ENABLE_CPU)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
        file(GLOB_RECURSE AKG_CPU_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
            "akg/cpu/*.cc"
        )
        list(APPEND AKG_SRC_LIST ${AKG_CPU_SRC_LIST})
    endif()
endif()

set_property(SOURCE ${KERNEL_SRC_LIST} ${CPU_SRC_LIST} ${GPU_SRC_LIST} ${D_SRC_LIST} ${AKG_SRC_LIST}
    PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_KERNEL)
add_library(_mindspore_backend_kernel_compiler_obj OBJECT ${KERNEL_SRC_LIST} ${CPU_SRC_LIST}
    ${GPU_SRC_LIST} ${D_SRC_LIST} ${AKG_SRC_LIST})
