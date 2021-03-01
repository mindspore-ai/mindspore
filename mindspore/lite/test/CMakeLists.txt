set(TOP_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../..)
set(TEST_DIR ${TOP_DIR}/mindspore/lite/test)
set(LITE_DIR ${TOP_DIR}/mindspore/lite)
set(CCSRC_DIR ${TOP_DIR}/mindspore/ccsrc)
set(CONVERTER_DIR ${TOP_DIR}/mindspore/lite/tools/converter)
include_directories(${TOP_DIR})
include_directories(${TEST_DIR})
include(${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/external_libs/gtest.cmake)

STRING(REPLACE " -fvisibility=hidden " " -fvisibility=default " CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
STRING(REPLACE " -fvisibility=hidden " " -fvisibility=default " CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

if(ENABLE_CONVERTER)
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
        ${LITE_DIR}/nnacl/*.c
        ${LITE_DIR}/nnacl/fp32/*.c
        ${LITE_DIR}/nnacl/int8/*.c
        ${LITE_DIR}/nnacl/base/*.c
        ${LITE_DIR}/nnacl/infer/*.c
        )

file(GLOB KERNEL_OP_TRAIN_SRC
        ${LITE_DIR}/nnacl/fp32_grad/*.c
        ${LITE_DIR}/src/runtime/kernel/arm/fp32_grad/*.cc
)

if(SUPPORT_TRAIN)
        list(APPEND KERNEL_OP_SRC ${KERNEL_OP_TRAIN_SRC})
endif()
if(PLATFORM_ARM64)
    # assembly
    file(GLOB TEST_ASSEMBLY_SRC ${LITE_DIR}/nnacl/assembly/arm64/*.s
            ${LITE_DIR}/nnacl/assembly/arm64/*.S)

    set_property(SOURCE ${TEST_ASSEMBLY_SRC} PROPERTY LANGUAGE C)
    set(KERNEL_OP_SRC
            ${KERNEL_OP_SRC}
            ${TEST_ASSEMBLY_SRC}
            )
endif()

if(PLATFORM_ARM32)
    # assembly
    file(GLOB TEST_ASSEMBLY_SRC
            ${LITE_DIR}/nnacl/assembly/arm32/*.S
            ${LITE_DIR}/nnacl/assembly/arm32/*.s)
    set_property(SOURCE ${TEST_ASSEMBLY_SRC} PROPERTY LANGUAGE C)
    set(KERNEL_OP_SRC
            ${KERNEL_OP_SRC}
            ${TEST_ASSEMBLY_SRC}
            )
endif()

if("${X86_64_SIMD}" STREQUAL "sse")
    file(GLOB TEST_ASSEMBLY_SRC ${LITE_DIR}/nnacl/intrinsics/sse/*.c)
    set_property(SOURCE ${TEST_ASSEMBLY_SRC} PROPERTY LANGUAGE C)
    set(KERNEL_OP_SRC
            ${KERNEL_OP_SRC}
            ${TEST_ASSEMBLY_SRC}
            )
endif()

if("${X86_64_SIMD}" STREQUAL "avx")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse4.1 -mavx -mavx2")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -msse4.1 -mavx -mavx2")
    file(GLOB TEST_ASSEMBLY_SRC ${LITE_DIR}/nnacl/intrinsics/sse/*.c
            ${LITE_DIR}/nnacl/intrinsics/avx/*.c
            ${LITE_DIR}/nnacl/assembly/avx/*.S)
    set_property(SOURCE ${TEST_ASSEMBLY_SRC} PROPERTY LANGUAGE C)
    set(KERNEL_OP_SRC
            ${KERNEL_OP_SRC}
            ${TEST_ASSEMBLY_SRC}
            )
endif()

### gpu kernel
if(SUPPORT_GPU STREQUAL opencl)
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
if(SUPPORT_GPU STREQUAL vulkan)
    file(GLOB GPU_KERNEL_OP_SRC
            ${LITE_DIR}/src/runtime/kernel/vulkan/kernel/*.cc
            )
    set(KERNEL_OP_SRC
            ${KERNEL_OP_SRC}
            ${GPU_KERNEL_OP_SRC}
            )
endif()

if(PLATFORM_ARM32 OR PLATFORM_ARM64)
  if(ENABLE_CONVERTER)
    set(BUILD_MINDDATA "off")
  endif()
endif()
### runtime framework
add_definitions(-DENABLE_V0)
file(GLOB_RECURSE OPS_SRC ${LITE_DIR}/src/ops/*.cc)
if(ENABLE_CONVERTER)
    set(OPS_SRC ${OPS_SRC})
endif()
set(TEST_LITE_SRC
        ${TEST_LITE_SRC}
        ${CCSRC_SRC}
        ${OPS_SRC}
        ${KERNEL_OP_SRC}
        ${LITE_DIR}/src/runtime/allocator.cc
        ${LITE_DIR}/src/runtime/runtime_api.cc
        ${LITE_DIR}/src/runtime/thread_pool.c
        ${LITE_DIR}/src/runtime/parallel_executor.cc
        ${LITE_DIR}/src/runtime/infer_manager.cc
        ${LITE_DIR}/src/tensor.cc
        ${LITE_DIR}/src/tensorlist.cc
        ${LITE_DIR}/src/executor.cc
        ${LITE_DIR}/src/inner_context.cc
        ${LITE_DIR}/src/kernel_registry.cc
        ${LITE_DIR}/src/lite_kernel.cc
        ${LITE_DIR}/src/lite_session.cc
        ${LITE_DIR}/src/dequant.cc
        ${LITE_DIR}/src/huffman_decode.cc
        ${LITE_DIR}/src/sub_graph_kernel.cc
        ${LITE_DIR}/src/lite_model.cc
        ${LITE_DIR}/src/scheduler.cc
        ${LITE_DIR}/src/common/graph_util.cc
        ${LITE_DIR}/src/common/prim_util.cc
        ${LITE_DIR}/src/common/tensor_util.cc
        ${LITE_DIR}/src/common/file_utils.cc
        ${LITE_DIR}/src/common/utils.cc
        ${LITE_DIR}/src/common/string_util.cc
        ${LITE_DIR}/tools/common/graph_util.cc
        ${LITE_DIR}/tools/common/tensor_util.cc
        ${LITE_DIR}/tools/common/node_util.cc
        ${LITE_DIR}/tools/common/flag_parser.cc
        ${LITE_DIR}/tools/common/storage.cc
        ${LITE_DIR}/tools/benchmark/benchmark.cc
        ${LITE_DIR}/test/st/benchmark_test.cc
        ${LITE_DIR}/src/errorcode.cc
        )
### gpu runtime
if(SUPPORT_GPU STREQUAL opencl)
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
if(SUPPORT_GPU STREQUAL vulkan)
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
        ${CORE_DIR}/mindrt/src/litebus.cc
        ${CORE_DIR}/mindrt/src/actor/actor.cc
        ${CORE_DIR}/mindrt/src/actor/actormgr.cc
        ${CORE_DIR}/mindrt/src/actor/actorpolicy.cc
        ${CORE_DIR}/mindrt/src/actor/actorthread.cc
        ${CORE_DIR}/mindrt/src/actor/aid.cc
        ${CORE_DIR}/mindrt/src/async/async.cc
        ${CORE_DIR}/mindrt/src/async/future.cc
        ${CORE_DIR}/mindrt/src/async/uuid_base.cc
        ${CORE_DIR}/mindrt/src/async/uuid_generator.cc
        )
endif()


### converter
if(ENABLE_CONVERTER)
    add_definitions(-DPRIMITIVE_WRITEABLE)
    add_definitions(-DUSE_GLOG)
    file(GLOB_RECURSE TEST_CASE_TFLITE_PARSERS_SRC
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
            ${LITE_DIR}/tools/optimizer/common/node_pass_extends.cc
            ${LITE_DIR}/tools/optimizer/common/pass_manager_extends.cc
            ${LITE_DIR}/tools/optimizer/common/gllo_utils.cc
            ${LITE_DIR}/tools/optimizer/fusion/conv_biasadd_fusion.cc
            ${LITE_DIR}/tools/optimizer/fusion/conv_activation_fusion.cc
            ${LITE_DIR}/tools/optimizer/fusion/conv_tuple_activation_fusion.cc
            ${LITE_DIR}/tools/optimizer/fusion/conv_transform_fusion.cc
            ${LITE_DIR}/tools/optimizer/fusion/conv_scale_fusion.cc
            ${LITE_DIR}/tools/optimizer/fusion/conv_bn_fusion.cc
            ${LITE_DIR}/tools/optimizer/fusion/constant_folding_fusion.cc
            ${LITE_DIR}/tools/optimizer/fusion/quant_dtype_cast_fusion.cc
            ${LITE_DIR}/tools/optimizer/fusion/tf_norm_fusion.cc
            ${LITE_DIR}/tools/optimizer/fusion/onnx_layer_norm_fusion.cc
            ${LITE_DIR}/tools/optimizer/fusion/batchmatmul_fusion.cc
            ${LITE_DIR}/tools/optimizer/fusion/sigmoid_mul_fusion.cc
            ${LITE_DIR}/tools/optimizer/fusion/conv_conv_fusion.cc
            ${LITE_DIR}/tools/optimizer/fusion/conv_tuplegetitem_fusion.cc
            ${LITE_DIR}/tools/optimizer/fusion/tflite_lstm_cell_fusion.cc
            ${LITE_DIR}/tools/optimizer/fusion/tf_lstm_cell_fusion.cc
            ${LITE_DIR}/tools/optimizer/fusion/tf_bidirection_gru_fusion.cc
            ${LITE_DIR}/tools/optimizer/fusion/tf_bidirection_gru_cf_fusion.cc
            ${LITE_DIR}/tools/optimizer/graph/weight_format_transform_pass.cc
            ${LITE_DIR}/tools/optimizer/graph/weight_format_hardcode_pass.cc
            ${LITE_DIR}/tools/optimizer/graph/clip_convert_activation_pass.cc
            ${LITE_DIR}/tools/optimizer/graph/group_depthwise_op_convert_pass.cc
            ${LITE_DIR}/tools/optimizer/graph/tflite_inputs_adjust_pass.cc
            ${LITE_DIR}/tools/optimizer/graph/update_conv2d_param_pass.cc
            ${LITE_DIR}/tools/optimizer/graph/unused_node_remove_pass.cc
            ${LITE_DIR}/tools/optimizer/graph/unused_cast_node_remove_pass.cc
            ${LITE_DIR}/tools/optimizer/graph/unused_transpose_node_remove_pass.cc
            ${LITE_DIR}/tools/optimizer/graph/redundant_op_remove_pass.cc
            ${LITE_DIR}/tools/optimizer/graph/infershape_pass.cc
            ${LITE_DIR}/tools/optimizer/graph/slice_prepose_pass.cc
            ${LITE_DIR}/tools/optimizer/graph/mindir_adjust_pass.cc
            ${LITE_DIR}/tools/optimizer/graph/onnx_inputs_adjust_pass.cc
            ${LITE_DIR}/tools/optimizer/graph/onnx_pad_adjust_pass.cc
            ${LITE_DIR}/tools/optimizer/graph/while_pass.cc
            ${LITE_DIR}/tools/optimizer/graph/if_pass.cc
            ${LITE_DIR}/tools/optimizer/graph/functionalize_control_op_pass.cc
            ${LITE_DIR}/tools/optimizer/graph/functionalize_while.cc
            ${LITE_DIR}/tools/optimizer/graph/functionalize_cond.cc
            ${LITE_DIR}/tools/optimizer/graph/inputs_adjust_pass.cc
            ${LITE_DIR}/tools/optimizer/graph/primitive_adjust_pass.cc
            )
endif()
### train
if(SUPPORT_TRAIN)
    set(TEST_LITE_SRC
            ${TEST_LITE_SRC}
            ${LITE_DIR}/src/train/train_populate_parameter.cc
            ${LITE_DIR}/src/train/train_populate_parameter_v0.cc
            ${LITE_DIR}/src/train/train_session.cc
            ${LITE_DIR}/src/train/transfer_session.cc
            ${LITE_DIR}/src/train/train_model.cc
            ${LITE_DIR}/src/lite_session.cc
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
        ${TEST_DIR}/ut/src/scheduler_test.cc
)

if(ENABLE_CONVERTER)
    set(TEST_SRC
            ${TEST_SRC}
            ${TEST_DIR}/st/converter_test.cc
            ${TEST_DIR}/st/control_flow_test.cc
            ${TEST_DIR}/st/sub_graph_test.cc
            ${TEST_DIR}/common/import_from_meta_graphT.cc
            ${TEST_DIR}/ut/tools/optimizer/fusion/conv_biasadd_fusion_test.cc
            ${TEST_DIR}/ut/tools/optimizer/fusion/conv_bn_fusion_test.cc
            ${TEST_DIR}/ut/tools/optimizer/fusion/conv_scale_fusion_test.cc
            ${TEST_DIR}/ut/tools/optimizer/fusion/conv_activation_fusion_test.cc
            ${TEST_DIR}/ut/tools/optimizer/fusion/constant_folding_fusion_test.cc
            )
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

if(SUPPORT_GPU STREQUAL opencl)
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

add_executable(lite-test ${TEST_SRC})
add_dependencies(lite-test fbs_src)

target_link_libraries(lite-test dl mindspore::gtest)

if(PLATFORM_ARM64 AND ENABLE_FP16)
    target_link_libraries(lite-test nnacl_fp16_mid nnacl_optimize_mid)
endif()

if(PLATFORM_ARM)
    target_link_libraries(lite-test log)
else()
    target_link_libraries(lite-test ${SECUREC_LIBRARY} pthread)
endif()

if(SUPPORT_NPU)
    include_directories(${DDK_PATH})
    target_link_libraries(lite-test npu_kernel_mid)
endif()
if(ENABLE_CONVERTER)
    add_dependencies(lite-test fbs_inner_src)
    target_link_libraries(lite-test
            anf_exporter_mid
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
