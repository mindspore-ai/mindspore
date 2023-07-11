# Compile ccsrc files in extendrt independently
string(REPLACE "-fvisibility-inlines-hidden" "" CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
string(REPLACE "-fvisibility=hidden" "" CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
string(REPLACE "-fvisibility-inlines-hidden" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
string(REPLACE "-fvisibility=hidden" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
if(MSLITE_ENABLE_CLOUD_FUSION_INFERENCE OR MSLITE_ENABLE_CLOUD_INFERENCE)
    if(PLATFORM_ARM64)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fexceptions")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fexceptions")
    endif()
    set(ENABLE_CPU on)
    set(LOAD_PLUGIN_STATIC on)
    string(REPLACE "-fno-rtti" "" CMAKE_C_FLAGS ${CMAKE_C_FLAGS})
    string(REPLACE "-fno-rtti" "" CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
    add_compile_definitions(ENABLE_CLOUD_FUSION_INFERENCE)
    add_compile_definitions(ENABLE_CLOUD_INFERENCE)
    remove_definitions(-DBUILD_LITE_INFERENCE)

    include_directories("${CCSRC_DIR}/ps/core")
    file(GLOB_RECURSE COMM_PROTO_IN RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "${CCSRC_DIR}/ps/core/protos/*.proto")
    ms_protobuf_generate(COMM_PROTO_SRCS COMM_PROTO_HDRS ${COMM_PROTO_IN})
    list(APPEND MSLITE_PROTO_SRC ${COMM_PROTO_SRCS})

    if(NOT ENABLE_SECURITY)
        include_directories("${CCSRC_DIR}/include/backend/debug/profiler/ascend")
        file(GLOB_RECURSE PROFILER_PROTO_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
                "${CCSRC_DIR}/plugin/device/ascend/hal/profiler/memory_profiling.proto")
        ms_protobuf_generate(PROFILER_MEM_PROTO_SRC PROFILER_MEM_PROTO_HDRS ${PROFILER_PROTO_LIST})
        list(APPEND MSLITE_PROTO_SRC ${PROFILER_MEM_PROTO_SRC})
    endif()

    include_directories("${CMAKE_BINARY_DIR}/runtime/graph_scheduler/actor/rpc")
    file(GLOB_RECURSE RPC_PROTO RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
            "${CCSRC_DIR}/runtime/graph_scheduler/actor/rpc/protocol/rpc.proto")
    ms_protobuf_generate(RPC_PROTO_SRCS RPC_PROTO_HDRS ${RPC_PROTO})
    list(APPEND MSLITE_PROTO_SRC ${RPC_PROTO_SRCS})

    include_directories(${TOP_DIR}/graphengine/910/inc/external/hccl)

    add_library(mindspore-lite-proto STATIC ${MSLITE_PROTO_SRC})

    set(ANF_ALG_SRC
            ${CCSRC_DIR}/utils/anfalgo.cc
            ${CCSRC_DIR}/utils/utils.cc
            ${CCSRC_DIR}/utils/parallel_context.cc
            ${CCSRC_DIR}/utils/convert_utils.cc)
    add_library(mindspore-infer-anfalgo OBJECT ${ANF_ALG_SRC})

    set(KERNEL_GRAPH_SRC
            ${CCSRC_DIR}/backend/common/session/kernel_graph.cc
            ${CCSRC_DIR}/backend/common/session/exec_order_builder.cc
            ${CCSRC_DIR}/backend/common/session/anf_runtime_algorithm.cc
            ${CCSRC_DIR}/backend/common/somas/somas.cc
            ${CCSRC_DIR}/backend/common/somas/somas_tensor.cc
            ${CCSRC_DIR}/backend/common/somas/somas_solver_pre.cc
            ${CCSRC_DIR}/backend/common/somas/somas_solver_core.cc
            ${CCSRC_DIR}/backend/common/somas/somas_solver_alg.cc
            ${CCSRC_DIR}/backend/operator/ops_backend_infer_function.cc
            ${CCSRC_DIR}/backend/graph_compiler/graph_partition.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/mock/segment_runner.cc
            ${CCSRC_DIR}/runtime/device/ms_device_shape_transfer.cc
            ${CCSRC_DIR}/runtime/device/kernel_info.cc
            ${CCSRC_DIR}/runtime/device/convert_tensor_utils.cc
            ${CCSRC_DIR}/runtime/device/kernel_runtime_manager.cc
            ${CCSRC_DIR}/runtime/device/kernel_runtime.cc
            ${CCSRC_DIR}/runtime/device/memory_scheduler.cc
            ${CCSRC_DIR}/runtime/device/memory_offload_strategy.cc
            ${CCSRC_DIR}/runtime/device/memory_manager.cc
            ${CCSRC_DIR}/runtime/device/auto_mem_offload.cc
            ${CCSRC_DIR}/runtime/device/gsm/mem_usage_analyzer.cc
            ${CCSRC_DIR}/runtime/device/gsm/swap_strategy_builder.cc
            ${CCSRC_DIR}/runtime/device/common_somas_allocator.cc
            ${CCSRC_DIR}/runtime/pynative/op_runtime_info.cc
            ${CCSRC_DIR}/runtime/hardware/device_type.cc
            ${CCSRC_DIR}/kernel/kernel_build_info.cc
            ${CCSRC_DIR}/kernel/ops_utils.cc
            ${CCSRC_DIR}/kernel/common_utils.cc
            ${CCSRC_DIR}/kernel/framework_utils.cc
            ${CCSRC_DIR}/kernel/philox_random.cc
            ${CCSRC_DIR}/kernel/kernel_factory.cc
            ${CCSRC_DIR}/kernel/kernel.cc
            ${CCSRC_DIR}/kernel/kernel_get_value.cc
            ${CCSRC_DIR}/kernel/kash/kernel_pack.cc
            ${CCSRC_DIR}/kernel/oplib/oplib.cc
            ${CCSRC_DIR}/kernel/oplib/super_bar.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/mock/anf_ir_dump.cc
            ${CCSRC_DIR}/common/debug/common.cc
            ${CCSRC_DIR}/common/debug/env_config_parser.cc
            ${CCSRC_DIR}/backend/common/mem_reuse/mem_dynamic_allocator.cc
            ${CCSRC_DIR}/common/thread_pool.cc
            ${CCSRC_DIR}/common/profiler.cc
            ${CCSRC_DIR}/utils/scoped_long_running.cc
            ${CCSRC_DIR}/utils/cse.cc
            ${CCSRC_DIR}/utils/comm_manager.cc
            ${CCSRC_DIR}/utils/signal_util.cc
            ${CORE_DIR}/utils/status.cc
            )

    add_library(mindspore-kernel-graph OBJECT ${KERNEL_GRAPH_SRC})
    add_dependencies(mindspore-kernel-graph fbs_src fbs_inner_src)
    add_dependencies(mindspore-kernel-graph mindspore-lite-proto)

    if(NOT PLATFORM_ARM)
        set(KERNEL_MOD_DEPEND_SRC
                ${CCSRC_DIR}/kernel/environ_manager.cc
                ${CCSRC_DIR}/utils/python_fallback_running.cc
                ${CCSRC_DIR}/runtime/device/tensors_queue.cc
                ${CCSRC_DIR}/runtime/device/tensor_array.cc
                ${CCSRC_DIR}/runtime/graph_scheduler/actor/actor_common.cc
                ${CCSRC_DIR}/runtime/hardware/device_context_manager.cc
                ${CCSRC_DIR}/plugin/device/cpu/hal/device/cpu_tensor_array.cc
                ${CCSRC_DIR}/plugin/device/cpu/hal/hardware/cpu_memory_pool.cc
                ${CCSRC_DIR}/distributed/embedding_cache/embedding_cache_utils.cc
                ${CCSRC_DIR}/distributed/embedding_cache/embedding_hash_map.cc
                ${CCSRC_DIR}/distributed/embedding_cache/embedding_storage/dense_embedding_storage.cc
                ${CCSRC_DIR}/distributed/embedding_cache/embedding_storage/sparse_embedding_storage.cc
                ${CCSRC_DIR}/distributed/embedding_cache/embedding_storage/embedding_storage.cc
                ${CCSRC_DIR}/distributed/persistent/storage/local_file.cc
                ${CCSRC_DIR}/distributed/persistent/storage/block.cc
                ${CCSRC_DIR}/distributed/persistent/storage/json_utils.cc
                ${CCSRC_DIR}/distributed/persistent/storage/file_io_utils.cc
                ${CCSRC_DIR}/distributed/cluster/dummy_cluster_context.cc
                ${CCSRC_DIR}/ps/ps_context.cc
                )
        add_library(_mindspore_cpu_kernel_mod_depend_obj OBJECT ${KERNEL_MOD_DEPEND_SRC})
        add_dependencies(_mindspore_cpu_kernel_mod_depend_obj fbs_src fbs_inner_src)
    endif()
endif()
