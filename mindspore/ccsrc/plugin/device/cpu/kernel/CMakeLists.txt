if(CMAKE_SYSTEM_NAME MATCHES "Darwin")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-delete-non-abstract-non-virtual-dtor -Wno-overloaded-virtual")
endif()

if(ENABLE_CPU)
    file(GLOB CPU_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        "*.cc"
        "eigen/*.cc"
        "mkldnn/*.cc"
        "fl/*.cc"
        "ps/*.cc"
        "pyfunc/*.cc"
        "rl/*.cc"
        "custom/*.cc"
        "environ/*.cc"
        "rpc/*.cc"
    )

    if(NOT ENABLE_MPI)
        list(REMOVE_ITEM CPU_SRC_LIST "allgather_cpu_kernel.cc")
        list(REMOVE_ITEM CPU_SRC_LIST "reduce_scatter_cpu_kernel.cc")
        list(REMOVE_ITEM CPU_SRC_LIST "embedding_look_up_comm_grad_cpu_kernel.cc")
    endif()

    if(PLATFORM_ARM64)
        add_compile_definitions(ENABLE_ARM)
    endif()
endif()

if(NOT ENABLE_CPU OR WIN32)
    list(REMOVE_ITEM CPU_SRC_LIST "ps/apply_momentum_ps_kernel.cc")
    list(REMOVE_ITEM CPU_SRC_LIST "ps/embedding_look_up_proxy_kernel.cc")
    list(REMOVE_ITEM CPU_SRC_LIST "ps/embedding_look_up_ps_kernel.cc")
    list(REMOVE_ITEM CPU_SRC_LIST "ps/pserver_kernel.cc")
    list(REMOVE_ITEM CPU_SRC_LIST "ps/pull_kernel.cc")
    list(REMOVE_ITEM CPU_SRC_LIST "ps/push_kernel.cc")
    list(REMOVE_ITEM CPU_SRC_LIST "ps/sparse_apply_adam_ps_kernel.cc")
    list(REMOVE_ITEM CPU_SRC_LIST "ps/sparse_apply_ftrl_ps_kernel.cc")
    list(REMOVE_ITEM CPU_SRC_LIST "ps/sparse_apply_lazy_adam_ps_kernel.cc")
    list(REMOVE_ITEM CPU_SRC_LIST "fl/fused_pull_weight_kernel.cc")
    list(REMOVE_ITEM CPU_SRC_LIST "fl/fused_push_weight_kernel.cc")
    list(REMOVE_ITEM CPU_SRC_LIST "fl/get_model_kernel.cc")
    list(REMOVE_ITEM CPU_SRC_LIST "fl/start_fl_job_kernel.cc")
    list(REMOVE_ITEM CPU_SRC_LIST "fl/update_model_kernel.cc")
    list(REMOVE_ITEM CPU_SRC_LIST "fl/push_metrics_kernel.cc")
    list(REMOVE_ITEM CPU_SRC_LIST "fl/get_keys_kernel.cc")
    list(REMOVE_ITEM CPU_SRC_LIST "fl/exchange_keys_kernel.cc")
endif()

if(ENABLE_AKG AND ${CMAKE_SYSTEM_NAME} MATCHES "Linux" AND ENABLE_CPU)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
    file(GLOB_RECURSE AKG_CPU_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        "akg/*.cc"
    )
endif()

set_property(SOURCE ${CPU_SRC_LIST} ${AKG_CPU_SRC_LIST}
        PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_KERNEL)
if(ENABLE_CPU)
    add_library(_mindspore_plugin_device_cpu_kernel_obj OBJECT ${CPU_SRC_LIST} ${AKG_CPU_SRC_LIST})
endif()
