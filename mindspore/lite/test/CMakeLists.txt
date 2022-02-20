set(TOP_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../..)
set(TEST_DIR ${TOP_DIR}/mindspore/lite/test)
set(LITE_DIR ${TOP_DIR}/mindspore/lite)
set(CCSRC_DIR ${TOP_DIR}/mindspore/ccsrc)
set(CONVERTER_DIR ${TOP_DIR}/mindspore/lite/tools/converter)
include_directories(${TOP_DIR})
include_directories(${TEST_DIR})
include_directories(${TOP_DIR}/mindspore/ccsrc/backend/kernel_compiler/cpu)
include(${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/external_libs/gtest.cmake)

STRING(REPLACE " -fvisibility=hidden " " -fvisibility=default " CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
STRING(REPLACE " -fvisibility=hidden " " -fvisibility=default " CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

if(MSLITE_ENABLE_CONVERTER)
    set(CCSRC_SRC
            ## ccsrc
            ${CCSRC_DIR}/backend/optimizer/common/pattern_engine.cc
            ${CCSRC_DIR}/backend/optimizer/common/visit.cc
            ${CCSRC_DIR}/backend/optimizer/common/optimizer.cc
            )
else()
    set(TEST_LITE_SRC ${LITE_DIR}/src/common/log_adapter.cc)
    add_compile_definitions(USE_ANDROID_LOG)
endif()

### cpu kernel
file(GLOB KERNEL_OP_SRC
        ${LITE_DIR}/src/runtime/kernel/arm/base/*.cc
        ${LITE_DIR}/src/runtime/kernel/arm/fp32/*.cc
        ${LITE_DIR}/src/runtime/kernel/arm/int8/*.cc
        ${LITE_DIR}/src/runtime/kernel/arm/string/*.cc
        )

if(NOT PLATFORM_ARM AND "${CMAKE_BUILD_TYPE}" STREQUAL "Release")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -fomit-frame-pointer -fstrict-aliasing -ffunction-sections \
        -fdata-sections -ffast-math")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -fomit-frame-pointer -fstrict-aliasing -ffunction-sections \
        -fdata-sections -ffast-math")
endif()

file(GLOB KERNEL_OP_TRAIN_SRC
        ${LITE_DIR}/src/runtime/kernel/arm/fp32_grad/*.cc
        )

file(GLOB FP16_KERNEL_OP_SRC
        ${LITE_DIR}/src/runtime/kernel/arm/fp16/*.cc
        )

if(SUPPORT_TRAIN)
    list(APPEND KERNEL_OP_SRC ${KERNEL_OP_TRAIN_SRC})
endif()

if(ENABLE_FP16)
    list(APPEND KERNEL_OP_SRC ${FP16_KERNEL_OP_SRC})
endif()

### gpu kernel
if(MSLITE_GPU_BACKEND STREQUAL opencl)
    file(GLOB GPU_KERNEL_OP_SRC
            ${LITE_DIR}/src/runtime/kernel/opencl/kernel/*.cc
            )
    set(KERNEL_OP_SRC
            ${KERNEL_OP_SRC}
            ${GPU_KERNEL_OP_SRC}
            ${LITE_DIR}/src/runtime/kernel/opencl/opencl_kernel.cc
            ${LITE_DIR}/src/runtime/kernel/opencl/opencl_subgraph.cc
            ${LITE_DIR}/src/runtime/kernel/opencl/opencl_fusion.cc
            ${LITE_DIR}/src/runtime/kernel/opencl/utils.cc
            )
endif()
if(MSLITE_GPU_BACKEND STREQUAL vulkan)
    file(GLOB GPU_KERNEL_OP_SRC
            ${LITE_DIR}/src/runtime/kernel/vulkan/kernel/*.cc
            )
    set(KERNEL_OP_SRC
            ${KERNEL_OP_SRC}
            ${GPU_KERNEL_OP_SRC}
            )
endif()

### runtime framework
add_definitions(-DENABLE_V0)

file(GLOB_RECURSE OPS_SRC ${LITE_DIR}/src/ops/*.cc)
file(GLOB CXX_SRC
        ${LITE_DIR}/src/c_api/*.cc
        ${LITE_DIR}/src/cxx_api/*.cc
        ${LITE_DIR}/src/cxx_api/graph/*.cc
        ${LITE_DIR}/src/cxx_api/model/*.cc
        ${LITE_DIR}/src/cxx_api/tensor/*.cc
        )
if(MSLITE_ENABLE_CONVERTER)
    set(OPS_SRC ${OPS_SRC})
endif()
set(TEST_LITE_SRC
        ${TEST_LITE_SRC}
        ${CCSRC_SRC}
        ${OPS_SRC}
        ${CXX_SRC}
        ${KERNEL_OP_SRC}
        ${LITE_DIR}/src/runtime/inner_allocator.cc
        ${LITE_DIR}/src/runtime/infer_manager.cc
        ${LITE_DIR}/src/tensor.cc
        ${LITE_DIR}/src/ms_tensor.cc
        ${LITE_DIR}/src/tensorlist.cc
        ${LITE_DIR}/src/executor.cc
        ${LITE_DIR}/src/inner_context.cc
        ${LITE_DIR}/src/kernel_registry.cc
        ${LITE_DIR}/src/inner_kernel.cc
        ${LITE_DIR}/src/lite_kernel.cc
        ${LITE_DIR}/src/lite_kernel_util.cc
        ${LITE_DIR}/src/lite_session.cc
        ${LITE_DIR}/src/weight_decoder.cc
        ${LITE_DIR}/src/huffman_decode.cc
        ${LITE_DIR}/src/sub_graph_kernel.cc
        ${LITE_DIR}/src/sub_graph_split.cc
        ${LITE_DIR}/src/lite_model.cc
        ${LITE_DIR}/src/scheduler.cc
        ${LITE_DIR}/src/common/graph_util.cc
        ${LITE_DIR}/src/common/prim_util.cc
        ${LITE_DIR}/src/common/tensor_util.cc
        ${LITE_DIR}/src/common/context_util.cc
        ${LITE_DIR}/src/common/file_utils.cc
        ${LITE_DIR}/src/common/utils.cc
        ${LITE_DIR}/src/common/dynamic_library_loader.cc
        ${LITE_DIR}/src/common/string_util.cc
        ${LITE_DIR}/src/common/quant_utils.cc
        ${LITE_DIR}/src/errorcode.cc
        ${LITE_DIR}/src/cpu_info.cc
        ${LITE_DIR}/tools/common/flag_parser.cc
        )

file(GLOB KERNEL_REG_SRC ${LITE_DIR}/src/registry/*.cc)
set(TEST_LITE_SRC ${TEST_LITE_SRC} ${KERNEL_REG_SRC})

if(ENABLE_TOOLS)
    set(TEST_LITE_SRC
        ${TEST_LITE_SRC}
        ${LITE_DIR}/tools/benchmark/benchmark.cc
        ${LITE_DIR}/test/st/benchmark_test.cc
        )
endif()

### gpu runtime
if(MSLITE_GPU_BACKEND STREQUAL opencl)
    include_directories(${CMAKE_BINARY_DIR}/_deps/opencl-headers-src)
    include_directories(${CMAKE_BINARY_DIR}/_deps/opencl-clhpp-src/include)
    file(GLOB_RECURSE OPENCL_RUNTIME_SRC
            ${LITE_DIR}/src/runtime/gpu/opencl/*.cc
            )
    set(TEST_LITE_SRC
            ${TEST_LITE_SRC}
            ${OPENCL_RUNTIME_SRC}
            )
endif()
if(MSLITE_GPU_BACKEND STREQUAL vulkan)
    include_directories(${LITE_DIR}/build/_deps/vulkan-headers-src/include)
    file(GLOB VULKAN_RUNTIME_SRC
            ${LITE_DIR}/src/runtime/gpu/*.cc
            ${LITE_DIR}/src/runtime/vulkan/*.cc
            )
    set(TEST_LITE_SRC
            ${TEST_LITE_SRC}
            ${VULKAN_RUNTIME_SRC}
            )
endif()

if(ENABLE_MINDRT)
    include_directories(${CORE_DIR}/mindrt/)
    include_directories(${CORE_DIR}/mindrt/src/)
    set(TEST_LITE_SRC ${TEST_LITE_SRC}
            ${LITE_DIR}/src/lite_mindrt.cc
            ${LITE_DIR}/src/mindrt_executor.cc
            ${CORE_DIR}/mindrt/src/mindrt.cc
            ${CORE_DIR}/mindrt/src/actor/actor.cc
            ${CORE_DIR}/mindrt/src/actor/actormgr.cc
            ${CORE_DIR}/mindrt/src/actor/actorpolicy.cc
            ${CORE_DIR}/mindrt/src/actor/aid.cc
            ${CORE_DIR}/mindrt/src/async/future.cc
            ${CORE_DIR}/mindrt/src/async/uuid_base.cc
            ${CORE_DIR}/mindrt/src/async/uuid_generator.cc
            ${CORE_DIR}/mindrt/src/thread/threadpool.cc
            ${CORE_DIR}/mindrt/src/thread/core_affinity.cc
            ${CORE_DIR}/mindrt/src/thread/actor_threadpool.cc
            )
endif()


### converter
if(MSLITE_ENABLE_CONVERTER)
    add_definitions(-DPRIMITIVE_WRITEABLE)
    add_definitions(-DUSE_GLOG)
    file(GLOB_RECURSE TEST_CASE_TFLITE_PARSERS_SRC
            ${TEST_DIR}/ut/tools/converter/registry/*.cc
            ${TEST_DIR}/ut/tools/converter/parser/tflite/*.cc
            )
    set(TEST_LITE_SRC
            ${TEST_LITE_SRC}
            ${TEST_CASE_TFLITE_PARSERS_SRC}
            ${LITE_DIR}/tools/converter/ops/while.cc
            ${LITE_DIR}/tools/common/protobuf_utils.cc
            ${LITE_DIR}/tools/converter/optimizer.cc
            ${LITE_DIR}/tools/converter/anf_transform.cc
            ${LITE_DIR}/tools/converter/graphdef_transform.cc
            ${LITE_DIR}/tools/converter/converter_flags.cc
            ${LITE_DIR}/tools/converter/converter.cc
            ${LITE_DIR}/tools/converter/export_model.cc
            ${LITE_DIR}/tools/converter/dump_graph.cc
            ${LITE_DIR}/tools/converter/parser/parser_utils.cc
            ${LITE_DIR}/tools/optimizer/common/node_pass_extends.cc
            ${LITE_DIR}/tools/optimizer/common/pass_manager_extends.cc
            ${LITE_DIR}/tools/optimizer/common/gllo_utils.cc
            ${LITE_DIR}/tools/optimizer/common/format_utils.cc
            ${LITE_DIR}/tools/optimizer/common/multiple_pattern_process_pass.cc
            ${LITE_DIR}/tools/optimizer/fusion/affine_activation_fusion.cc
            ${LITE_DIR}/tools/optimizer/fusion/affine_fusion.cc
            ${LITE_DIR}/tools/optimizer/fusion/conv_biasadd_fusion.cc
            ${LITE_DIR}/tools/optimizer/fusion/conv_activation_fusion.cc
            ${LITE_DIR}/tools/optimizer/fusion/conv_tuple_activation_fusion.cc
            ${LITE_DIR}/tools/optimizer/fusion/conv_transform_fusion.cc
            ${LITE_DIR}/tools/optimizer/fusion/conv_scale_fusion.cc
            ${LITE_DIR}/tools/optimizer/fusion/conv_bn_fusion.cc
            ${LITE_DIR}/tools/optimizer/fusion/multi_head_attention_fusion.cc
            ${LITE_DIR}/tools/optimizer/fusion/reshape_reshape_fusion.cc
            ${LITE_DIR}/tools/optimizer/fusion/constant_folding_fusion.cc
            ${LITE_DIR}/tools/optimizer/fusion/norm_fusion.cc
            ${LITE_DIR}/tools/optimizer/fusion/batchmatmul_fusion.cc
            ${LITE_DIR}/tools/optimizer/fusion/sigmoid_mul_fusion.cc
            ${LITE_DIR}/tools/optimizer/fusion/conv_conv_fusion.cc
            ${LITE_DIR}/tools/optimizer/fusion/conv_tuplegetitem_fusion.cc
            ${LITE_DIR}/tools/optimizer/fusion/tflite_lstm_cell_fusion.cc
            ${LITE_DIR}/tools/optimizer/fusion/tf_lstm_cell_fusion.cc
            ${LITE_DIR}/tools/optimizer/fusion/tf_bidirection_gru_fusion.cc
            ${LITE_DIR}/tools/optimizer/fusion/matmul_add_fusion.cc
            ${LITE_DIR}/tools/optimizer/fusion/mul_add_fusion.cc
            ${LITE_DIR}/tools/optimizer/fusion/tflite_rel_pos_multi_head_attention_fusion.cc
            ${LITE_DIR}/tools/optimizer/fusion/glu_fusion.cc
            ${LITE_DIR}/tools/optimizer/fusion/gelu_fusion.cc
            ${LITE_DIR}/tools/optimizer/fusion/tf_gelu_fusion.cc
            ${LITE_DIR}/tools/optimizer/fusion/onnx_gelu_fusion.cc
            ${LITE_DIR}/tools/optimizer/fusion/squeeze_fusion.cc
            ${LITE_DIR}/tools/optimizer/graph/add_tensor_array.cc
            ${LITE_DIR}/tools/optimizer/graph/conv1d_weight_expanding_pass.cc
            ${LITE_DIR}/tools/optimizer/graph/clip_convert_activation_pass.cc
            ${LITE_DIR}/tools/optimizer/graph/group_depthwise_op_convert_pass.cc
            ${LITE_DIR}/tools/optimizer/graph/update_conv2d_param_pass.cc
            ${LITE_DIR}/tools/optimizer/graph/unused_cast_node_remove_pass.cc
            ${LITE_DIR}/tools/optimizer/graph/unused_transpose_node_remove_pass.cc
            ${LITE_DIR}/tools/optimizer/graph/redundant_op_remove_pass.cc
            ${LITE_DIR}/tools/optimizer/graph/infershape_pass.cc
            ${LITE_DIR}/tools/optimizer/graph/slice_prepose_pass.cc
            ${LITE_DIR}/tools/optimizer/graph/control_flow_pass.cc
            ${LITE_DIR}/tools/optimizer/graph/unify_format_pass.cc
            ${LITE_DIR}/tools/optimizer/graph/node_infershape.cc
            ${LITE_DIR}/tools/optimizer/graph/transpose_strategy.cc
            ${LITE_DIR}/tools/optimizer/graph/reduce_same_act_pass.cc
            ${LITE_DIR}/tools/optimizer/graph/split_one_pass.cc
            ${LITE_DIR}/tools/optimizer/fisson/eliminate_concat_split.cc
            ${LITE_DIR}/tools/optimizer/fisson/fisson_util.cc
            ${LITE_DIR}/tools/optimizer/fisson/iter_node_outputs.cc
            ${LITE_DIR}/tools/optimizer/fisson/node_out_shapes.cc
            ${LITE_DIR}/tools/optimizer/fisson/multi_conv_split_pass.cc
            ${LITE_DIR}/tools/optimizer/parallel/multi_node_split.cc
            ${LITE_DIR}/tools/optimizer/parallel/multi_conv_info.cc
            ${LITE_DIR}/tools/optimizer/parallel/parallel_pass.cc
            ${LITE_DIR}/tools/optimizer/parallel/operator_info.cc
            ${LITE_DIR}/tools/optimizer/parallel/operator_info_register.cc
            ${LITE_DIR}/tools/optimizer/parallel/conv2d_info.cc
            ${LITE_DIR}/tools/optimizer/parallel/depthwise_conv2d_info.cc
            ${LITE_DIR}/tools/optimizer/parallel/spliter.cc
            ${LITE_DIR}/tools/optimizer/parallel/split_strategy.cc
            ${LITE_DIR}/tools/common/graph_util.cc
            ${LITE_DIR}/tools/common/meta_graph_utils.cc
            ${LITE_DIR}/tools/common/tensor_util.cc
            ${LITE_DIR}/tools/common/node_util.cc
            ${LITE_DIR}/tools/common/storage.cc
            ${LITE_DIR}/tools/converter/parser/inputs_adjust.cc
            ${LITE_DIR}/tools/converter/parser/unused_node_remove_pass.cc
            ${LITE_DIR}/tools/converter/parser/conv1d_inout_adjust.cc
            ${LITE_DIR}/tools/converter/parser/tf_bidirection_gru_cf_fusion.cc
            ${LITE_DIR}/tools/converter/import/mindspore_importer.cc
            ${LITE_DIR}/tools/converter/import/primitive_adjust.cc
            ${LITE_DIR}/tools/converter/import/mindir_adjust.cc
            )
else()
set(TEST_LITE_SRC
        ${TEST_LITE_SRC}
        ${LITE_DIR}/tools/common/meta_graph_utils.cc
        ${LITE_DIR}/tools/converter/optimizer.cc
        ${LITE_DIR}/tools/converter/legacy_optimizer/graph/dropout_node_remove_pass.cc
        ${LITE_DIR}/tools/converter/legacy_optimizer/graph/isolated_node_remove_pass.cc
        ${LITE_DIR}/tools/converter/legacy_optimizer/graph/subgraph_node_pass.cc
        )
endif()
### train
if(SUPPORT_TRAIN)
    set(TEST_LITE_SRC
            ${TEST_LITE_SRC}
            ${LITE_DIR}/src/train/train_populate_parameter.cc
            ${LITE_DIR}/src/train/train_populate_parameter_v0.cc
            ${LITE_DIR}/src/train/train_session.cc
            ${LITE_DIR}/src/train/train_export.cc
            ${LITE_DIR}/src/train/train_utils.cc
            ${LITE_DIR}/src/train/transfer_session.cc
            ${LITE_DIR}/src/train/graph_dropout.cc
            ${LITE_DIR}/src/lite_session.cc
            ${LITE_DIR}/tools/common/storage.cc
            )
else()
    set(TEST_LITE_SRC
            ${TEST_LITE_SRC}
            ${LITE_DIR}/src/train/train_populate_parameter.cc
            ${LITE_DIR}/src/train/train_populate_parameter_v0.cc
            ${LITE_DIR}/src/lite_session.cc
            )
endif()
### test src
file(GLOB_RECURSE TEST_CASE_KERNEL_SRC
        ${TEST_DIR}/ut/src/runtime/kernel/arm/common/*.cc
        ${TEST_DIR}/ut/src/runtime/kernel/arm/fp32/*.cc
        ${TEST_DIR}/ut/src/runtime/kernel/arm/int8/*.cc
        ${TEST_DIR}/ut/src/runtime/kernel/arm/string/*.cc
        ${TEST_DIR}/ut/nnacl/infer/*.cc
        )

file(GLOB_RECURSE TEST_CASE_KERNEL_TRAIN_SRC
        ${TEST_DIR}/ut/src/runtime/kernel/arm/fp32_grad/*.cc
        )

set(TEST_SRC
        ${TEST_LITE_SRC}
        ${TEST_MINDDATA_SRC}
        ${TEST_CASE_KERNEL_SRC}
        ${TEST_DIR}/main.cc
        ${TEST_DIR}/common/common_test.cc
        ${TEST_DIR}/ut/src/infer_test.cc
        ${TEST_DIR}/ut/src/utils_test.cc
        ${TEST_DIR}/ut/src/lite_mindrt_test.cc
        ${TEST_DIR}/ut/src/dynamic_library_loader_test.cc
        ${TEST_DIR}/ut/src/scheduler_test.cc
        ${TEST_DIR}/ut/src/lite_mindrt_test.cc
        ${TEST_DIR}/ut/src/registry/registry_test.cc
        ${TEST_DIR}/ut/src/registry/registry_custom_op_test.cc
        ${TEST_DIR}/ut/src/api/context_c_test.cc
        ${TEST_DIR}/ut/src/api/tensor_c_test.cc
        )

if(MSLITE_ENABLE_CONVERTER)
    set(TEST_SRC
            ${TEST_SRC}
            ${TEST_DIR}/st/converter_test.cc
            ${TEST_DIR}/st/mindrt_parallel_test.cc
            ${TEST_DIR}/st/graph_test.cc
            ${TEST_DIR}/st/sub_graph_test.cc
            ${TEST_DIR}/common/import_from_meta_graphT.cc
            ${TEST_DIR}/ut/tools/optimizer/fusion/conv_biasadd_fusion_test.cc
            ${TEST_DIR}/ut/tools/optimizer/fusion/conv_bn_fusion_test.cc
            ${TEST_DIR}/ut/tools/optimizer/fusion/conv_scale_fusion_test.cc
            ${TEST_DIR}/ut/tools/optimizer/fusion/conv_activation_fusion_test.cc
            ${TEST_DIR}/ut/tools/optimizer/fusion/constant_folding_fusion_test.cc
            )
else()
    set(TEST_SRC
        ${TEST_SRC}
        ${CORE_DIR}/utils/status.cc)
endif()

if(SUPPORT_TRAIN)
    set(TEST_SRC
            ${TEST_SRC}
            ${TEST_CASE_KERNEL_TRAIN_SRC}
            ${TEST_DIR}/ut/src/infer_test.cc  # temporary
            )
else()
    set(TEST_SRC
            ${TEST_SRC}
            ${TEST_DIR}/ut/src/infer_test.cc
            )
endif()

if(MSLITE_GPU_BACKEND STREQUAL opencl)
    file(GLOB_RECURSE TEST_CASE_KERNEL_GPU_SRC
            ${TEST_DIR}/ut/src/runtime/kernel/opencl/*.cc
            )
    set(TEST_SRC
            ${TEST_SRC}
            ${TEST_CASE_KERNEL_GPU_SRC}
            )
endif()

if(ENABLE_FP16)
    file(GLOB_RECURSE TEST_CASE_KERNEL_FP16_SRC
            ${TEST_DIR}/ut/src/runtime/kernel/arm/fp16/*.cc
            )
    set(TEST_SRC
            ${TEST_SRC}
            ${TEST_CASE_KERNEL_FP16_SRC}
            )
endif()

if(ENABLE_FP16 AND SUPPORT_TRAIN)
    file(GLOB_RECURSE TEST_CASE_KERNEL_FP16_SRC_GRAD
            ${TEST_DIR}/ut/src/runtime/kernel/arm/fp16_grad/*.cc)
    list(APPEND TEST_SRC ${TEST_CASE_KERNEL_FP16_SRC_GRAD})
endif()

file(GLOB_RECURSE API_SRC ${LITE_DI}/src/cxx_api/*.cc)
set(TEST_SRC ${TEST_SRC} ${API_SRC})
add_executable(lite-test ${TEST_SRC})
add_dependencies(lite-test fbs_src)

if(SUPPORT_TRAIN)
    add_dependencies(lite-test fbs_inner_src)
endif()

target_link_libraries(lite-test
        nnacl_mid
        dl
        mindspore::gtest
        )

if(PLATFORM_ARM AND ENABLE_FP16)
    target_link_libraries(lite-test nnacl_fp16_mid)
    if(PLATFORM_ARM64)
        target_link_libraries(lite-test nnacl_optimize_mid)
    endif()
endif()

if(PLATFORM_ARM)
    target_link_libraries(lite-test log)
else()
    target_link_libraries(lite-test ${SECUREC_LIBRARY} pthread)
endif()

if(SUPPORT_NPU)
    include_directories(${DDK_PATH})
    target_link_libraries(lite-test npu_kernel_mid c++_shared)
endif()
if(MSLITE_ENABLE_CONVERTER)
    add_dependencies(lite-test fbs_inner_src)
    target_link_libraries(lite-test
            anf_exporter_mid
            mslite_converter_plugin
            tflite_parser_mid
            caffe_parser_mid
            onnx_parser_mid
            tf_parser_mid
            graph_pass_mid
            fusion_mid
            quantizer_mid
            proto_mid
            mindspore::protobuf
            mindspore::eigen
            mindspore::json
            -Wl,--whole-archive mindspore_core -Wl,--no-whole-archive
            mindspore::glog
            )
endif()
if(ENABLE_MODEL_OBF)
    target_link_libraries(lite-test ${OBF_LIB_DIR}/libmsdeobfuscator-lite.so)
endif()
