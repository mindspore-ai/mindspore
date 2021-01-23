file(GLOB_RECURSE KERNEL_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
    "kernel_build_info.cc"
    "kash/*.cc"
    "common_utils.cc"
    "oplib/*.cc"
)

if(CMAKE_SYSTEM_NAME MATCHES "Darwin")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-delete-non-abstract-non-virtual-dtor")
endif()

if(ENABLE_D)
    file(GLOB_RECURSE D_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        "kernel_query.cc"
        "kernel_fusion.cc"
        "akg/ascend/*.cc"
        "akg/akg_kernel_json_generator.cc"
        "akg/akg_kernel_json_decoder.cc"
        "akg/akg_kernel_attrs_process.cc"
        "akg/akg_kernel_metadata.cc"
        "tbe/*.cc"
        "host/*.cc"
        "aicpu/*.cc"
        "rts/*.cc"
        "hccl/*.cc"
    )
    add_compile_definitions(ENABLE_D)
endif()

if(ENABLE_CPU)
    file(GLOB_RECURSE CPU_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        "cpu/*.cc"
    )

    if(NOT ENABLE_MPI)
        list(REMOVE_ITEM CPU_SRC_LIST "cpu/allgather_cpu_kernel.cc")
        list(REMOVE_ITEM CPU_SRC_LIST "cpu/reduce_scatter_cpu_kernel.cc")
        list(REMOVE_ITEM CPU_SRC_LIST "cpu/embedding_look_up_comm_grad_cpu_kernel.cc")
    endif()
endif()

if(NOT (ENABLE_CPU AND (ENABLE_D OR ENABLE_GPU)))
    list(REMOVE_ITEM CPU_SRC_LIST "cpu/ps/apply_momentum_ps_kernel.cc")
    list(REMOVE_ITEM CPU_SRC_LIST "cpu/ps/embedding_look_up_proxy_kernel.cc")
    list(REMOVE_ITEM CPU_SRC_LIST "cpu/ps/embedding_look_up_ps_kernel.cc")
    list(REMOVE_ITEM CPU_SRC_LIST "cpu/ps/pserver_kernel.cc")
    list(REMOVE_ITEM CPU_SRC_LIST "cpu/ps/pull_kernel.cc")
    list(REMOVE_ITEM CPU_SRC_LIST "cpu/ps/push_kernel.cc")
    list(REMOVE_ITEM CPU_SRC_LIST "cpu/ps/sparse_apply_adam_ps_kernel.cc")
    list(REMOVE_ITEM CPU_SRC_LIST "cpu/ps/sparse_apply_ftrl_ps_kernel.cc")
    list(REMOVE_ITEM CPU_SRC_LIST "cpu/ps/sparse_apply_lazy_adam_ps_kernel.cc")
endif()

if(ENABLE_GPU)
    file(GLOB_RECURSE CUDA_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        "gpu/*.cu"
        "akg/gpu/*.cc"
        "akg/akg_kernel_json_generator.cc"
        "akg/akg_kernel_json_decoder.cc"
        "akg/akg_kernel_attrs_process.cc"
    )

    file(GLOB_RECURSE GPU_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "gpu/*.cc")
    list(REMOVE_ITEM GPU_SRC_LIST "gpu/nccl/nccl_collective_gpu_kernel.cc")
    list(REMOVE_ITEM GPU_SRC_LIST "gpu/nccl/nccl_send_gpu_kernel.cc")
    list(REMOVE_ITEM GPU_SRC_LIST "gpu/nccl/nccl_recv_gpu_kernel.cc")

    if(ENABLE_MPI)
        include(ExternalProject)
        file(GLOB_RECURSE GPU_NCCL_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "gpu/nccl/*.cc")
        list(APPEND GPU_SRC_LIST ${GPU_NCCL_LIST})
    endif()

    # add_library(_mindspore_kernel_cuda_obj OBJECT ${CUDA_SRC_LIST})
endif()

set_property(SOURCE ${KERNEL_SRC_LIST} ${CPU_SRC_LIST} ${GPU_SRC_LIST} ${D_SRC_LIST}
    PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_KERNEL)
add_library(_mindspore_backend_kernel_compiler_obj OBJECT ${KERNEL_SRC_LIST} ${CPU_SRC_LIST} ${GPU_SRC_LIST} ${D_SRC_LIST})
