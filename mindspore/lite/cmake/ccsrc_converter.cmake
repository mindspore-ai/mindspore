# Compile ccsrc files in converter independently
if(MSLITE_ENABLE_CONVERTER)
    add_definitions(-DPRIMITIVE_WRITEABLE)
    add_definitions(-DUSE_GLOG)
    set(USE_GLOG on)
    if(MSLITE_ENABLE_MODEL_ENCRYPTION)
        add_compile_definitions(ENABLE_OPENSSL)
    endif()

    if(ENABLE_GPU)
        add_compile_definitions(ENABLE_GPU)
    endif()

    set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../src)
    set(TOOLS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../tools)
    set(CCSRC_SRC
            ${CCSRC_DIR}/backend/common/optimizer/pattern_engine.cc
            ${CCSRC_DIR}/backend/common/optimizer/visitor.cc
            ${CCSRC_DIR}/backend/common/optimizer/graph_optimizer.cc
            ${CCSRC_DIR}/backend/operator/ops_backend_infer_function.cc
            ${CCSRC_DIR}/kernel/kernel_factory.cc
            )

    if(MSLITE_ENABLE_CLOUD_FUSION_INFERENCE OR MSLITE_ENABLE_CLOUD_INFERENCE)
        set(CCSRC_SRC ${CCSRC_SRC}
                ${CCSRC_DIR}/ps/ps_context.cc
                ${CCSRC_DIR}/common/thread_pool.cc
                ${CCSRC_DIR}/common/profiler.cc
                ${CCSRC_DIR}/plugin/device/cpu/kernel/cpu_kernel.cc
                ${CCSRC_DIR}/distributed/cluster/dummy_cluster_context.cc
                ${CCSRC_DIR}/kernel/ops_utils.cc
                ${CCSRC_DIR}/kernel/common_utils.cc
                ${CCSRC_DIR}/kernel/framework_utils.cc
                ${CCSRC_DIR}/kernel/philox_random.cc
                ${CCSRC_DIR}/kernel/kash/kernel_pack.cc
                ${CCSRC_DIR}/kernel/kernel_build_info.cc
                ${CCSRC_DIR}/kernel/oplib/oplib.cc
                ${CCSRC_DIR}/kernel/kernel.cc
                ${CCSRC_DIR}/kernel/kernel_get_value.cc
                ${CCSRC_DIR}/kernel/oplib/super_bar.cc
                ${CCSRC_DIR}/runtime/device/kernel_info.cc
                ${CCSRC_DIR}/runtime/graph_scheduler/actor/actor_common.cc
                ${CCSRC_DIR}/runtime/device/ms_device_shape_transfer.cc
                ${CCSRC_DIR}/runtime/hardware/device_type.cc
                ${CCSRC_DIR}/runtime/device/kernel_runtime_manager.cc
                ${CCSRC_DIR}/runtime/hardware/device_context_manager.cc
                ${CCSRC_DIR}/runtime/device/convert_tensor_utils.cc
                ${CCSRC_DIR}/utils/comm_manager.cc
                ${CCSRC_DIR}/backend/common/session/exec_order_builder.cc
                ${CCSRC_DIR}/backend/common/session/kernel_graph.cc
                ${CCSRC_DIR}/backend/common/session/anf_runtime_algorithm.cc
                ${SRC_DIR}/extendrt/utils/tensor_utils.cc
                )
    endif()

    if(NOT WIN32)
        set(CCSRC_SRC ${CCSRC_SRC}
                ${CCSRC_DIR}/utils/anfalgo.cc
                ${CCSRC_DIR}/utils/convert_utils.cc
                ${CCSRC_DIR}/utils/utils.cc
                ${CCSRC_DIR}/utils/parallel_context.cc
                )
    endif()

    if(ENABLE_GPU)
        add_compile_definitions(ENABLE_GPU)
    endif()

    if(MSLITE_ENABLE_GRAPH_KERNEL)

        if(AKG_USE_LLVM)
            add_compile_definitions(AKG_USE_LLVM)
            message(STATUS "Converter support Graph Kernel CPU backend")
        endif()

        if(AKG_ENABLE_D)
            add_compile_definitions(AKG_ENABLE_D)
            message(STATUS "Converter support Graph Kernel Ascend backend")
        endif()

        if(AKG_USE_CUDA)
            add_compile_definitions(AKG_USE_CUDA)
            message(STATUS "Converter support Graph Kernel CUDA backend")
        endif()

        add_compile_definitions(MSLITE_ENABLE_GRAPH_KERNEL)
        file(GLOB_RECURSE GRAPH_KERNEL_SRC
                ${TOOLS_DIR}/graph_kernel/common/*.cc
                ${TOOLS_DIR}/graph_kernel/converter/*.cc
                ${CCSRC_DIR}/backend/common/graph_kernel/core/*.cc
                ${CCSRC_DIR}/backend/common/graph_kernel/expander/*.cc
                ${CCSRC_DIR}/backend/common/graph_kernel/expanders/*.cc
                ${CCSRC_DIR}/backend/common/graph_kernel/model/*.cc
                ${CCSRC_DIR}/backend/common/graph_kernel/split_model/*.cc
                ${CCSRC_DIR}/backend/common/graph_kernel/graph_kernel_flags.cc
                ${CCSRC_DIR}/kernel/graph_kernel/graph_kernel_json_generator.cc
                )
        set(CCSRC_SRC
                ${CCSRC_SRC}
                ${GRAPH_KERNEL_SRC}
                )
    endif()

    set_property(SOURCE ${CCSRC_SRC} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_LITE)
    add_library(ccsrc_src_mid OBJECT ${CCSRC_SRC})
    add_dependencies(ccsrc_src_mid fbs_src fbs_inner_src)
    if(MSLITE_ENABLE_CLOUD_INFERENCE)
        add_dependencies(ccsrc_src_mid mindspore-lite-proto)
    endif()
    target_compile_definitions(ccsrc_src_mid PRIVATE BACKEND_DLL)
endif()
