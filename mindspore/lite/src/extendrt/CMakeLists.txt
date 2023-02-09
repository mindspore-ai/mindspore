if(MSLITE_ENABLE_CLOUD_FUSION_INFERENCE OR MSLITE_ENABLE_CLOUD_INFERENCE)
    if(PLATFORM_ARM64)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fexceptions")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fexceptions")
    endif()
    set(ENABLE_CPU on)
    set(LOAD_PLUGIN_STATIC on)
    add_compile_definitions(USE_GLOG)
    string(REPLACE "-fno-rtti" "" CMAKE_C_FLAGS ${CMAKE_C_FLAGS})
    string(REPLACE "-fno-rtti" "" CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
    add_compile_definitions(ENABLE_CLOUD_FUSION_INFERENCE)
    add_compile_definitions(ENABLE_CLOUD_INFERENCE)
    remove_definitions(-DBUILD_LITE_INFERENCE)

    set(MSLITE_KERNEL_PLUGIN
        ${MSLITE_KERNEL_PLUGIN}
        ${CMAKE_CURRENT_SOURCE_DIR}/kernel/ascend/plugin/ascend_kernel_plugin.cc)

    set(MSLITE_EXTEND_RUNTIME_SRC
        ${MSLITE_KERNEL_PLUGIN}
        ${CMAKE_CURRENT_SOURCE_DIR}/../common/file_utils.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../common/utils.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../common/graph_util.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../common/config_infos.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../common/config_file.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/subgraph_kernel.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/numa_adapter.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/kernel/cpu/less_test_kernel_mod.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/kernel/cpu/transpose_kernel_mod.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/infer_session.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/session/single_op_session.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/infer_device_address.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/utils/kernel_build_utils.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/utils/kernel_graph_utils.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/utils/tensor_utils.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/utils/runtime_utils.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/utils/serialization.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/utils/func_graph_utils.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/session/delegate_session.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/session/factory.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/delegate/factory.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/delegate/plugin/tensorrt_executor_plugin.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/delegate/plugin/litert_executor_plugin.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/delegate/plugin/ascend_ge_executor_plugin.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/delegate/tensorrt/distribution/distribution_base.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/delegate_graph_executor.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/session/optimizer/tensorrt_optimizer.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/delegate/graph_executor/litert/func_graph_reuse_manager.cc
        )
    if(MSLITE_ENABLE_BFC_MEMORY)
        set(MSLITE_EXTEND_RUNTIME_SRC ${MSLITE_EXTEND_RUNTIME_SRC}
            ${CMAKE_CURRENT_SOURCE_DIR}/dynamic_mem_allocator.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/dynamic_mem_manager.cc
            )
    endif()
    include_directories("${CCSRC_DIR}/ps/core")
    file(GLOB_RECURSE COMM_PROTO_IN RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "${CCSRC_DIR}/ps/core/protos/*.proto")
    ms_protobuf_generate(COMM_PROTO_SRCS COMM_PROTO_HDRS ${COMM_PROTO_IN})
    list(APPEND MSLITE_PROTO_SRC ${COMM_PROTO_SRCS})

    if(NOT ENABLE_SECURITY)
        include_directories("${CCSRC_DIR}/profiler/device/ascend")
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

    add_library(mindspore-lite-proto STATIC ${MSLITE_PROTO_SRC})

    set(ANF_ALG_SRC ${ANF_ALG_SRC}
        ${CCSRC_DIR}/utils/anfalgo.cc
        ${CCSRC_DIR}/utils/utils.cc
        ${CCSRC_DIR}/utils/parallel_context.cc
        ${CCSRC_DIR}/utils/convert_utils.cc)
    add_library(mindspore-infer-anfalgo OBJECT ${ANF_ALG_SRC})

    set(KERNEL_GRAPH_SRC ${KERNEL_GRAPH_SRC}
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
        ${CCSRC_DIR}/runtime/device/common_somas_allocator.cc
        ${CCSRC_DIR}/runtime/pynative/op_runtime_info.cc
        ${CCSRC_DIR}/runtime/hardware/device_type.cc
        ${CCSRC_DIR}/kernel/kernel_build_info.cc
        ${CCSRC_DIR}/kernel/common_utils.cc
        ${CCSRC_DIR}/kernel/kernel.cc
        ${CCSRC_DIR}/kernel/kash/kernel_pack.cc
        ${CCSRC_DIR}/kernel/oplib/oplib.cc
        ${CCSRC_DIR}/kernel/oplib/super_bar.cc
        ${CCSRC_DIR}/common/debug/anf_dump_utils.cc
        ${CCSRC_DIR}/common/debug/anf_ir_dump.cc
        ${CCSRC_DIR}/common/debug/common.cc
        ${CCSRC_DIR}/common/debug/env_config_parser.cc
        ${CCSRC_DIR}/common/mem_reuse/mem_dynamic_allocator.cc
        ${CCSRC_DIR}/common/thread_pool.cc
        ${CCSRC_DIR}/utils/scoped_long_running.cc
        ${CCSRC_DIR}/utils/cse.cc
        ${CCSRC_DIR}/utils/comm_manager.cc
        ${CCSRC_DIR}/utils/signal_util.cc
        ${CORE_DIR}/utils/status.cc
        )
    add_library(mindspore-kernel-graph OBJECT ${KERNEL_GRAPH_SRC})

    add_library(mindspore-extendrt SHARED ${MSLITE_EXTEND_RUNTIME_SRC})
    add_dependencies(mindspore-extendrt fbs_inner_src)
    add_dependencies(mindspore-extendrt mindspore-infer-anfalgo)
    add_dependencies(mindspore-extendrt mindspore-kernel-graph)
    add_subdirectory(cxx_api)
    add_subdirectory(delegate/graph_executor/litert)
    add_library(mindspore-extendrt_static STATIC ${MSLITE_EXTEND_RUNTIME_SRC})
    add_dependencies(mindspore-extendrt_static fbs_inner_src)
    add_dependencies(mindspore-extendrt_static mindspore-infer-anfalgo)
    add_dependencies(mindspore-extendrt_static mindspore-kernel-graph)

    add_subdirectory(${CCSRC_DIR}/backend/common/pass common_pass)
    add_subdirectory(${CCSRC_DIR}/backend/operator backend_operator)
    add_subdirectory(${CCSRC_DIR}/backend/common/optimizer mindspore_ccsrc_backend_cmmon_optimizer)
    target_link_libraries(mindspore-extendrt mindspore_infer_shared_lib_obj)
    target_link_libraries(mindspore-extendrt mindspore-infer-anfalgo
        mindspore-kernel-graph _mindspore_backend_common_optimizer_obj
        _mindspore_backend_common_pass_obj)
    target_link_libraries(mindspore-extendrt mindspore_core mindspore::protobuf)
    target_link_libraries(mindspore-extendrt_static mindspore_infer_shared_lib_obj)
    target_link_libraries(mindspore-extendrt_static mindspore-infer-anfalgo
            mindspore-kernel-graph _mindspore_backend_common_optimizer_obj
            _mindspore_backend_common_pass_obj _mindspore_backend_operator_obj)
    target_link_libraries(mindspore-extendrt_static mindspore_core mindspore::protobuf)
    add_dependencies(mindspore-extendrt_static msplugin-ge-litert)
    target_link_libraries(mindspore-extendrt_static msplugin-ge-litert)

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
        ${CCSRC_DIR}/distributed/embedding_cache/embedding_storage/embedding_storage.cc
        ${CCSRC_DIR}/distributed/persistent/storage/local_file.cc
        ${CCSRC_DIR}/distributed/persistent/storage/block.cc
        ${CCSRC_DIR}/distributed/persistent/storage/json_utils.cc
        ${CCSRC_DIR}/distributed/persistent/storage/file_io_utils.cc
        ${CCSRC_DIR}/distributed/cluster/dummy_cluster_context.cc
        ${CCSRC_DIR}/ps/ps_context.cc
        )
    add_library(_mindspore_cpu_kernel_mod_depend_obj OBJECT ${KERNEL_MOD_DEPEND_SRC})
    target_link_libraries(mindspore-extendrt _mindspore_cpu_kernel_mod_depend_obj
            mindspore-lite-proto mindspore::dnnl)
    target_link_libraries(mindspore-extendrt_static _mindspore_cpu_kernel_mod_depend_obj
            mindspore-lite-proto mindspore::dnnl)

    set(CPU_KERNEL_OBJECT_COUNT 0)
    add_subdirectory(${CCSRC_DIR}/plugin/device/cpu/kernel lite_kernel_mod)
    foreach(number RANGE 1 ${CPU_KERNEL_OBJECT_COUNT})
        target_link_libraries(mindspore-extendrt _mindspore_plugin_device_cpu_kernel_obj_${number})
        target_link_libraries(mindspore-extendrt_static _mindspore_plugin_device_cpu_kernel_obj_${number})
    endforeach()
    endif()

    if(NOT WIN32)
        target_link_libraries(mindspore-extendrt dl)
        target_link_libraries(mindspore-extendrt_static dl)
    endif()

    if(MSLITE_ENABLE_ACL)
        include_directories(${TOP_DIR}/graphengine/inc/external)
        add_subdirectory(kernel/ascend)
        add_subdirectory(delegate/ascend_ge)
    endif()

    if(SUPPORT_CUDA)
        set(CUDA_PATH $ENV{CUDA_HOME})
        include_directories(${CCSRC_DIR}/plugin/device/gpu/kernel)
        set(ENABLE_GPU on)
        add_definitions(-DENABLE_GPU)
        set(CUDA_VERSION 11.1)
        include_directories(${CUDA_PATH})
        include_directories(${CUDA_PATH}/include)
        find_package(CUDA)
        add_subdirectory(kernel/cuda)
        list(APPEND CUDA_NVCC_FLAGS -arch=sm_53 --expt-relaxed-constexpr)
        add_subdirectory(${CCSRC_DIR}/plugin/device/gpu/kernel/cuda_impl/cuda_ops cuda_ops)
        target_link_libraries(mindspore-extendrt cuda_lite_kernel_mid cuda_ops)
        target_link_libraries(mindspore-extendrt_static cuda_lite_kernel_mid cuda_ops)
    endif()

    if(SUPPORT_TENSORRT)
        add_definitions(-DSUPPORT_TENSORRT)
        add_subdirectory(delegate/tensorrt)
    endif()

    if(MSLITE_ENABLE_CONVERTER)
        add_subdirectory(convert)
        target_link_libraries(mindspore-extendrt -Wl,--no-as-needed mindspore_converter)
    endif()

    set(TEST_CLOUD_INFER off)

    if(TEST_CLOUD_INFER)
        set(CLOUD_INFER_TEST_SRC ${CLOUD_INFER_TEST_SRC}
            ${CMAKE_CURRENT_SOURCE_DIR}/utils/cloud_infer_test.cc
            ${LITE_DIR}/tools/common/flag_parser.cc
            )

        add_executable(cloud_infer_test ${CLOUD_INFER_TEST_SRC})
        add_dependencies(cloud_infer_test mindspore-extendrt)
        target_link_libraries(cloud_infer_test mindspore-extendrt)
    endif()
else()
    set(MSLITE_EXTEND_RUNTIME_SRC ${MODEL_LOADER_FRAMEWORK_SRC})
    add_library(mindspore-extendrt OBJECT ${MSLITE_EXTEND_RUNTIME_SRC})
    add_dependencies(mindspore-extendrt fbs_inner_src)
    add_library(mindspore-extendrt_static OBJECT ${MSLITE_EXTEND_RUNTIME_SRC})
    add_dependencies(mindspore-extendrt_static fbs_inner_src)
endif()

set_target_properties(mindspore-extendrt PROPERTIES OUTPUT_NAME "mindspore-lite")

