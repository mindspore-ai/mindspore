set(API_SRC "")
set(LITE_SRC "")

add_compile_definitions(ENABLE_LITE_HELPER)

set(MOCK_SRC
        ${MOCK_SRC}
        ${LITE_DIR}/src/extendrt/mock/lite_runtime/converters.cc
        ${LITE_DIR}/src/extendrt/mock/lite_runtime/populate/base_operator_populate_register.cc
        ${LITE_DIR}/src/extendrt/mock/lite_runtime/populate/arithmetic_populate.cc
        )

set(LITE_SRC
        ${LITE_SRC}
        ${API_SRC}
        ${MOCK_SRC}
        ${LITE_DIR}/src/common/context_util.cc
        ${LITE_DIR}/src/common/file_utils.cc
        ${LITE_DIR}/src/common/utils.cc
        ${LITE_DIR}/src/common/graph_util.cc
        ${LITE_DIR}/src/common/log.cc
        ${LITE_DIR}/src/common/prim_util.cc
        ${LITE_DIR}/src/common/tensor_util.cc
        ${LITE_DIR}/src/common/config_infos.cc
        ${LITE_DIR}/src/litert/allocator.cc
        ${LITE_DIR}/src/litert/inner_allocator.cc
        ${LITE_DIR}/src/litert/runtime_allocator.cc
        ${LITE_DIR}/src/litert/infer_manager.cc
        ${LITE_DIR}/src/litert/runtime_shape_fusion_pass.cc
        ${LITE_DIR}/src/litert/runtime_pass.cc
        # ${LITE_DIR}/src/litert/pass/runtime_ncx_pass.cc
        ${LITE_DIR}/src/litert/schema_tensor_wrapper.cc
        ${LITE_DIR}/src/tensor.cc
        ${LITE_DIR}/src/tensorlist.cc
        ${LITE_DIR}/src/litert/tensor_category.cc
        ${LITE_DIR}/src/litert/executor.cc
        ${LITE_DIR}/src/litert/inner_context.cc
        ${LITE_DIR}/src/litert/lite_model.cc
        ${LITE_DIR}/src/litert/kernel_registry.cc
        ${LITE_DIR}/src/litert/lite_kernel.cc
        ${LITE_DIR}/src/litert/kernel_exec.cc
        ${LITE_DIR}/src/litert/kernel_exec_util.cc
        ${LITE_DIR}/src/litert/sub_graph_kernel.cc
        ${LITE_DIR}/src/litert/scheduler.cc
        ${LITE_DIR}/src/litert/lite_session.cc
        ${LITE_DIR}/src/errorcode.cc
        ${LITE_DIR}/src/litert/cpu_info.cc
        ${LITE_DIR}/src/litert/pack_weight_manager.cc
        ${LITE_DIR}/src/control_flow/control_flow_scheduler.cc
        ${LITE_DIR}/src/control_flow/control_subgraph_creator.cc
        ${LITE_DIR}/src/extendrt/utils/tensor_utils.cc
        )

set(MODEL_LOADER_FRAMEWORK_SRC
        ${LITE_DIR}/src/extendrt/mindir_loader/model_loader.cc
        )

set(MINDIR_MODEL_SRC
        ${MINDIR_MODEL_SRC}
        ${LITE_DIR}/src/extendrt/mindir_loader/mindir_model/mindir_model.cc
        ${LITE_DIR}/src/extendrt/mindir_loader/mindir_model/mindir_model_util.cc
        ${LITE_DIR}/src/extendrt/mindir_loader/mindir_model/mindir_model_convertor.cc
        ${LITE_DIR}/src/extendrt/mindir_loader/mindir_model/mindir_model_loader.cc
        ${LITE_DIR}/src/extendrt/mindir_loader/mindir_model/kernel_mod_util.cc
        )

set(MINDIR_KERNEL_SRC
        ${MINDIR_KERNEL_SRC}
        ${LITE_DIR}/src/extendrt/mindir_loader/mindir_model/inner_kernel.cc)

set(LITE_SRC
        ${LITE_SRC}
        ${MODEL_LOADER_FRAMEWORK_SRC}
        ${MINDIR_MODEL_SRC}
        ${MINDIR_KERNEL_SRC}
        )

set(INFER_HELPER_SRC
        ${INFER_HELPER_SRC}
        ${LITE_DIR}/src/common/helper/external_tensor/file_helper.cc
        ${LITE_DIR}/src/common/helper/external_tensor/memory_helper.cc
        )

set(LITE_SRC
        ${LITE_SRC}
        ${INFER_HELPER_SRC}
        )

if(MSLITE_ENABLE_MODEL_ENCRYPTION)
    set(LITE_SRC
            ${LITE_SRC}
            ${LITE_DIR}/src/common/decrypt.cc
            ${LITE_DIR}/src/common/dynamic_library_loader.cc
            )
endif()

if(MSLITE_ENABLE_BFC_MEMORY)
    set(LITE_SRC
        ${LITE_SRC}
        ${LITE_DIR}/src/extendrt/dynamic_mem_allocator.cc
        ${LITE_DIR}/src/extendrt/dynamic_mem_manager.cc
        ${LITE_DIR}/src/extendrt/numa_adapter.cc
        )
endif()

if(MSLITE_ENABLE_SHARING_MODEL_WEIGHT)
    set(LITE_SRC
        ${LITE_SRC}
        ${LITE_DIR}/src/litert/pack_weight.cc
        )
endif()

if(MSLITE_ENABLE_DYNAMIC_THREAD_DISTRIBUTE)
    set(LITE_SRC
        ${LITE_SRC}
        ${LITE_DIR}/src/litert/thread_cost_model.cc
        )
endif()

if(MSLITE_ENABLE_CONTROLFLOW)
    file(GLOB CONTROL_FLOW_KERNEL_SRC
            ${LITE_DIR}/src/control_flow/kernel/*.cc
            )
    set(LITE_SRC ${LITE_SRC} ${CONTROL_FLOW_KERNEL_SRC})
endif()

if(MSLITE_ENABLE_RUNTIME_GLOG)
    if(NOT MSLITE_ENABLE_RUNTIME_CONVERT AND NOT MSLITE_ENABLE_KERNEL_EXECUTOR
            AND NOT (MSLITE_ENABLE_CLOUD_FUSION_INFERENCE OR MSLITE_ENABLE_CLOUD_INFERENCE))
        set(LITE_SRC ${LITE_SRC}
            ${CORE_DIR}/utils/log_adapter.cc)
    endif()
endif()

if(MSLITE_ENABLE_RUNTIME_CONVERT)
    file(GLOB RUNTIME_CONVERT_SRC
            ${LITE_DIR}/src/common/ops/ops_def.cc
            ${LITE_DIR}/src/common/ops/ops_utils.cc
            ${LITE_DIR}/src/common/ops/anf_utils.cc
            ${LITE_DIR}/src/litert/runtime_convert.cc)
    set(LITE_SRC ${LITE_SRC} ${RUNTIME_CONVERT_SRC})
endif()

if(MSLITE_ENABLE_STRING_KERNEL)
    set(LITE_SRC
        ${LITE_SRC}
        ${LITE_DIR}/src/common/string_util.cc
        )
endif()

if(ENABLE_RUNTIME_NCX_PASS)
    file(GLOB RUNTIME_NCX_PASS_SRC
            ${LITE_DIR}/src/litert/pass/decrease_transpose_algo.cc
            ${LITE_DIR}/src/litert/pass/delete_isolated_kernel.cc
            ${LITE_DIR}/src/litert/pass/infershape_pass.cc
            ${LITE_DIR}/src/litert/pass/pass_utils.cc
            ${LITE_DIR}/src/litert/pass/runtime_optimizer.cc
            ${LITE_DIR}/src/litert/pass/to_nchw_format.cc
            ${LITE_DIR}/src/litert/pass/transpose_strategy.cc
            )
    set(LITE_SRC ${LITE_SRC} ${RUNTIME_NCX_PASS_SRC})
endif()

if(MSLITE_ENABLE_WEIGHT_DECODE)
    set(LITE_SRC ${LITE_SRC}
            ${LITE_DIR}/src/litert/huffman_decode.cc)
    if(NOT MSLITE_ENABLE_RUNTIME_CONVERT)
        set(LITE_SRC ${LITE_SRC}
            ${TOOLS_DIR}/converter/quantizer/fse_decoder.cc
            ${TOOLS_DIR}/converter/quantizer/fse_bit_stream.cc)
    endif()
endif()

if(MSLITE_ENABLE_RUNTIME_PASS)
    file(GLOB KERNEL_ONLINE_FUSION_SRC ${LITE_DIR}/src/litert/pass/online_fusion/*.cc)
    set(LITE_SRC
        ${LITE_SRC}
        ${KERNEL_ONLINE_FUSION_SRC}
        ${LITE_DIR}/src/litert/sub_graph_split.cc
    )
endif()

if(MSLITE_ENABLE_AUTO_PARALLEL)
    set(LITE_SRC
            ${LITE_SRC}
            ${LITE_DIR}/src/litert/sub_graph_split.cc
            )
endif()

if(MSLITE_ENABLE_CUSTOM_KERNEL)
    file(GLOB KERNEL_REG_SRC ${LITE_DIR}/src/registry/*.cc)
endif()

set(LITE_SRC
    ${LITE_SRC}
    ${KERNEL_REG_SRC}
    ${LITE_DIR}/src/litert/weight_decoder.cc
    )

if(MSLITE_GPU_BACKEND STREQUAL opencl)
    file(GLOB_RECURSE OPENCL_RUNTIME_SRC
            ${LITE_DIR}/src/litert/kernel/gpu/opencl/*.cc
            )
    set(LITE_SRC
            ${LITE_SRC}
            ${OPENCL_RUNTIME_SRC}
            )
endif()

if(MSLITE_GPU_BACKEND STREQUAL cuda)
    file(GLOB CUDA_RUNTIME_SRC
            ${LITE_DIR}/src/litert/gpu/*.cc
            ${LITE_DIR}/src/litert/gpu/cuda/*.cc
            )
    set(LITE_SRC
            ${LITE_SRC}
            ${CUDA_RUNTIME_SRC}
            )
endif()
if(MSLITE_MINDDATA_IMPLEMENT STREQUAL "full")
  set(TRAIN_SRC_WITH_MD
            ${LITE_DIR}/src/train/train_loop.cc
          )
endif()


file(GLOB CXX_API_EXPRESSION
  ${LITE_DIR}/src/litert/cxx_api/expression/*.cc
  )

file(GLOB EXPRESSION_OPS
  ${LITE_DIR}/src/expression/ops/*.cc
  )

set(EXPRESSION_SRC
    ${CXX_API_EXPRESSION}
    ${LITE_DIR}/src/expression/export.cc
    ${LITE_DIR}/src/expression/expr.cc
    ${LITE_DIR}/src/expression/import.cc
    ${LITE_DIR}/src/expression/net.cc
    ${LITE_DIR}/src/expression/node.cc
    ${LITE_DIR}/src/expression/ops.cc
    ${LITE_DIR}/src/expression/ops_utils.cc
    ${LITE_DIR}/src/expression/param.cc
    ${LITE_DIR}/src/expression/sequential.cc
    ${EXPRESSION_OPS}
  )

set(TRAIN_SRC
        ${API_TRAIN_SRC}
        ${TRAIN_SRC_WITH_MD}
        ${EXPRESSION_SRC}
        ${LITE_DIR}/src/common/quant_utils.cc
        ${LITE_DIR}/src/litert/cxx_api/metrics/accuracy.cc
        ${LITE_DIR}/src/litert/cxx_api/train/model_build.cc
        ${LITE_DIR}/src/litert/cxx_api/train/model_build_impl.cc
        ${LITE_DIR}/src/litert/cxx_api/train/converters.cc
        ${LITE_DIR}/src/litert/cxx_api/train/train_support.cc
        ${LITE_DIR}/src/train/train_populate_parameter.cc
        ${LITE_DIR}/src/train/train_session.cc
        ${LITE_DIR}/src/train/graph_fusion.cc
        ${LITE_DIR}/src/train/graph_dropout.cc
        ${LITE_DIR}/src/train/transfer_session.cc
        ${LITE_DIR}/src/train/train_utils.cc
        ${LITE_DIR}/src/train/loss_monitor.cc
        ${LITE_DIR}/src/train/lr_scheduler.cc
        ${LITE_DIR}/src/train/accuracy_metrics.cc
        ${LITE_DIR}/src/train/accuracy_monitor.cc
        ${LITE_DIR}/src/train/classification_train_accuracy_monitor.cc
        ${LITE_DIR}/src/train/train_export.cc
        ${LITE_DIR}/src/train/opt_allocator.cc
        ${LITE_DIR}/src/common/storage.cc
        ${TOOLS_DIR}/converter/optimizer.cc
        ${TOOLS_DIR}/converter/legacy_optimizer/fusion/fusion_pass.cc
        ${TOOLS_DIR}/converter/legacy_optimizer/fusion/fusion_pattern.cc
        ${TOOLS_DIR}/common/meta_graph_utils.cc
        ${TOOLS_DIR}/converter/legacy_optimizer/fusion/matmul_biasadd_fusion_pass.cc
        ${TOOLS_DIR}/converter/legacy_optimizer/graph/dropout_node_remove_pass.cc
        ${TOOLS_DIR}/converter/legacy_optimizer/graph/isolated_node_remove_pass.cc
        ${TOOLS_DIR}/converter/legacy_optimizer/graph/subgraph_node_pass.cc
        )

if(MSLITE_ENABLE_MINDRT)
    set(LITE_SRC ${LITE_SRC}
        ${LITE_DIR}/src/litert/lite_mindrt.cc
        ${LITE_DIR}/src/litert/mindrt_executor.cc
        ${LITE_DIR}/src/control_flow/control_actor_creator.cc
        )
    if(MSLITE_ENABLE_CONTROLFLOW)
        file(GLOB CONTROL_FLOW_ACTOR_SRC
                ${LITE_DIR}/src/control_flow/actor/*.cc
                )
        set(LITE_SRC ${LITE_SRC} ${CONTROL_FLOW_ACTOR_SRC})
    endif()
else()
    set(LITE_SRC ${LITE_SRC}
        ${CORE_DIR}/mindrt/src/thread/core_affinity.cc
        ${CORE_DIR}/mindrt/src/thread/threadpool.cc
        )
endif()

if(MSLITE_ENABLE_GRAPH_KERNEL)
    file(GLOB_RECURSE GRAPH_KERNEL_SRC
        ${TOOLS_DIR}/graph_kernel/common/*.cc
        ${TOOLS_DIR}/graph_kernel/runtime/*.cc
        )
    set(LITE_SRC ${LITE_SRC} ${GRAPH_KERNEL_SRC})
endif()

if(NOT MSLITE_ENABLE_COREML)
    set(LITE_SRC ${LITE_SRC}
        ${LITE_DIR}/src/litert/delegate/coreml/stub/coreml_delegate_stub.cc)
endif()

set(MSLITE_GE_LITERT_SRC
    ${MSLITE_GE_LITERT_SRC}
    ${LITE_DIR}/src/extendrt/delegate/graph_executor/litert/graph_executor.cc
    ${LITE_DIR}/src/extendrt/delegate/graph_executor/litert/litert_plugin_impl.cc
    )

set(LITE_SRC ${LITE_SRC} ${MSLITE_GE_LITERT_SRC})

add_library(unified_runtime_lite_src_mid OBJECT ${LITE_SRC})
add_dependencies(unified_runtime_lite_src_mid fbs_src)

add_library(lite-unified-executor SHARED $<TARGET_OBJECTS:unified_runtime_lite_src_mid>)

add_dependencies(lite-unified-executor mindspore_converter)
target_link_libraries(lite-unified-executor mindspore_converter)

if(MSVC)
    set_target_properties(lite-unified-executor PROPERTIES PREFIX lib)
    set_target_properties(lite-unified-executor PROPERTIES IMPORT_PREFIX lib)
    set_target_properties(lite-unified-executor PROPERTIES IMPORT_SUFFIX .dll.lib)
endif()

target_link_libraries(lite-unified-executor cpu_kernel_mid nnacl_mid cpu_ops_mid)

if(SUPPORT_TRAIN)
    target_link_libraries(lite-unified-executor train_cpu_kernel_mid)
endif()

if(MSLITE_ENABLE_MINDRT)
    target_link_libraries(lite-unified-executor mindrt_mid)
endif()

if(PLATFORM_ARM32 OR PLATFORM_ARM64 AND NOT TARGET_HIMIX
        AND NOT TARGET_MIX210 AND NOT TARGET_OHOS_LITE AND NOT MACHINE_LINUX_ARM64)
    target_link_libraries(lite-unified-executor log)
endif()
if(MSLITE_MINDDATA_IMPLEMENT STREQUAL "lite")
    target_link_libraries(lite-unified-executor minddata_eager_mid minddata-lite)
endif()

########################## build optimize and float16 library #################################
if(PLATFORM_ARM)
    if(PLATFORM_ARM64 AND NOT TARGET_HIMIX AND NOT TARGET_MIX210 AND NOT MACHINE_LINUX_ARM64)
        target_link_libraries(lite-unified-executor cpu_opt_kernel_mid nnacl_optimize_mid)
    endif()
    if(MSLITE_ENABLE_FP16)
        target_link_libraries(lite-unified-executor cpu_fp16_kernel_mid nnacl_fp16_mid)
    endif()
endif()

if(MSLITE_ENABLE_RUNTIME_CONVERT)
    target_link_libraries(lite-unified-executor
            quantizer_mid fusion_mid proto_mid graph_pass_mid preprocess_mid
            cpu_kernel_mid ccsrc_src_mid converter_src_mid lite_exporter_mid
            config_parser_mid mslite_converter_plugin mindspore_core coder_mid
            mindir_serializer_mid mindspore::protobuf ${SECUREC_LIBRARY})
    target_link_libraries(lite-unified-executor
            ccsrc_debug_common_mid_ mindir_proto_mid _mindspore_transform_express_ir_obj)
endif()


target_link_libraries(lite-unified-executor mindspore::protobuf)

if(MSLITE_ENABLE_RUNTIME_GLOG)
    target_link_libraries(lite-unified-executor mindspore::glog)
endif()

if(NOT WIN32)
    target_link_libraries(lite-unified-executor dl)
endif()

if(ENABLE_MODEL_OBF)
    target_link_libraries(lite-unified-executor ${OBF_LIB_DIR}/libmsdeobfuscator-lite.so)
endif()
