set(TOP_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../..)
set(TEST_DIR ${TOP_DIR}/mindspore/lite/test)
set(LITE_DIR ${TOP_DIR}/mindspore/lite)
set(CCSRC_DIR ${TOP_DIR}/mindspore/ccsrc)
include_directories(${TOP_DIR})
include_directories(${TEST_DIR})
include(${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/dependency_gtest.cmake)

### anf src
set(ANF_SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/../../core/ir/meta_tensor.cc
        ${CORE_DIR}/gvar/logging_level.cc
        ${CORE_DIR}/gvar/typeid_manager.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../../core/base/base.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../../core/utils/log_adapter.cc
        )

if(BUILD_CONVERTER)
    set(ANF_SRC
            ${ANF_SRC}
            # core/base
            ${CMAKE_CURRENT_SOURCE_DIR}/../../core/base/base_ref.cc
            # core/ir
            ${CMAKE_CURRENT_SOURCE_DIR}/../../core/ir/anf.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/../../core/ir/anf_extends.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/../../core/ir/meta_func_graph.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/../../core/ir/func_graph.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/../../core/ir/graph_utils.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/../../core/ir/func_graph_cloner.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/../../core/ir/func_graph_extends.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/../../core/ir/manager.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/../../core/ir/primitive.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/../../core/ir/tensor.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/../../core/ir/visitor.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/../../core/ir/meta_tensor_extends.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/../../core/ir/dtype.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/../../core/ir/dtype_extends.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/../../core/ir/named.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/../../core/ir/scope.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/../../core/ir/value.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/../../core/ir/value_extends.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/../../core/ir/dtype/container.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/../../core/ir/dtype/empty.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/../../core/ir/dtype/number.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/../../core/ir/dtype/ref.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/../../core/ir/dtype/type.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/../../core/ir/dtype/type_extends.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/../../core/utils/any.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/../../core/utils/symbolic.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/../../core/utils/misc.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/../../core/utils/trace_base.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/../../core/utils/trace_info.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/../../core/utils/label.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/../../core/utils/info.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/../../core/utils/profile.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/../../core/utils/ms_context.cc
            # core/abstract
            ${CMAKE_CURRENT_SOURCE_DIR}/../../core/abstract/abstract_function.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/../../core/abstract/analysis_context.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/../../core/abstract/param_validator.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/../../core/abstract/abstract_value.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/../../core/abstract/dshape.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/../../core/abstract/utils.cc
            ## ccsrc
            ${CCSRC_DIR}/debug/draw.cc
            ${CCSRC_DIR}/pybind_api/export_flags.cc
            ${CCSRC_DIR}/utils/context/context_extends.cc
            ${CCSRC_DIR}/frontend/parallel/costmodel_context.cc
            ${CCSRC_DIR}/backend/optimizer/common/pattern_engine.cc
            ${CCSRC_DIR}/backend/optimizer/common/visit.cc
            ${CCSRC_DIR}/backend/optimizer/common/optimizer.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/../src/common/graph_utils_extends.cc
            )
else()
    set(ANF_SRC
            ${ANF_SRC}
            ${CMAKE_CURRENT_SOURCE_DIR}/../src/ir/meta_tensor_extends.cc
            )
endif()
### cpu kernel
file(GLOB KERNEL_OP_SRC
        ${LITE_DIR}/src/runtime/kernel/arm/base/*.cc
        ${LITE_DIR}/src/runtime/kernel/arm/fp32/*.cc
        ${LITE_DIR}/src/runtime/kernel/arm/int8/*.cc
        ${LITE_DIR}/src/runtime/kernel/arm/nnacl/*.c
        ${LITE_DIR}/src/runtime/kernel/arm/nnacl/fp32/*.c
        ${LITE_DIR}/src/runtime/kernel/arm/nnacl/int8/*.c
        ${LITE_DIR}/src/runtime/kernel/arm/nnacl/quantization/*.c
        )

file(GLOB KERNEL_OP_TRAIN_SRC
        ${LITE_DIR}/src/runtime/kernel/arm/nnacl/fp32_grad/*.c
        ${LITE_DIR}/src/runtime/kernel/arm/fp32_grad/*.cc
)

if (SUPPORT_TRAIN)
        list(APPEND KERNEL_OP_SRC ${KERNEL_OP_TRAIN_SRC})
endif()

if (PLATFORM_ARM64)
    # assembly
    file(GLOB TEST_ASSEMBLY_SRC ${LITE_DIR}/src/runtime/kernel/arm/nnacl/assembly/arm64/*.s
            ${LITE_DIR}/src/runtime/kernel/arm/nnacl/assembly/arm64/*.S)

    set_property(SOURCE ${TEST_ASSEMBLY_SRC} PROPERTY LANGUAGE C)
    set(KERNEL_OP_SRC
            ${KERNEL_OP_SRC}
            ${TEST_ASSEMBLY_SRC}
            )
endif()
#[[
if (PLATFORM_ARM32)
    # assembly
    file(GLOB TEST_ASSEMBLY_SRC
            ${LITE_DIR}/src/runtime/kernel/arm/nnacl/assembly/arm32/*.S
            ${LITE_DIR}/src/runtime/kernel/arm/nnacl/assembly/arm32/*.s)
    set_property(SOURCE ${TEST_ASSEMBLY_SRC} PROPERTY LANGUAGE C)
    set(KERNEL_OP_SRC
            ${KERNEL_OP_SRC}
            ${TEST_ASSEMBLY_SRC}
            )
endif()
]]
if (ENABLE_FP16)
    file(GLOB KERNEL_OP_FP16_SRC
            ${LITE_DIR}/src/runtime/kernel/arm/fp16/*.cc
            ${LITE_DIR}/src/runtime/kernel/arm/nnacl/fp16/*.c
            )
    set(KERNEL_OP_SRC
            ${KERNEL_OP_SRC}
            ${KERNEL_OP_FP16_SRC}
            )
endif ()
### gpu kernel
if (SUPPORT_GPU)
    set(KERNEL_OP_SRC
            ${KERNEL_OP_SRC}
            ${LITE_DIR}/src/runtime/kernel/opencl/subgraph_opencl_kernel.cc
            ${LITE_DIR}/src/runtime/kernel/opencl/utils.cc
            ${LITE_DIR}/src/runtime/kernel/opencl/kernel/arithmetic.cc
            ${LITE_DIR}/src/runtime/kernel/opencl/kernel/convolution.cc
            ${LITE_DIR}/src/runtime/kernel/opencl/kernel/depthwise_conv2d.cc
            ${LITE_DIR}/src/runtime/kernel/opencl/kernel/pooling2d.cc
            ${LITE_DIR}/src/runtime/kernel/opencl/kernel/matmul.cc
            ${LITE_DIR}/src/runtime/kernel/opencl/kernel/softmax.cc
            ${LITE_DIR}/src/runtime/kernel/opencl/kernel/concat.cc
            ${LITE_DIR}/src/runtime/kernel/opencl/kernel/activation.cc
            ${LITE_DIR}/src/runtime/kernel/opencl/kernel/conv2d_transpose.cc
            ${LITE_DIR}/src/runtime/kernel/opencl/kernel/transpose.cc
            ${LITE_DIR}/src/runtime/kernel/opencl/kernel/reshape.cc
            )
endif()
### minddata lite
if (BUILD_MINDDATA)
    include_directories(${CCSRC_DIR}/minddata)
    set(DATASET_TEST_DIR ${TEST_DIR}/ut/src/dataset)
    set(TEST_MINDDATA_SRC
                ${DATASET_TEST_DIR}/de_tensor_test.cc
                ${DATASET_TEST_DIR}/eager_test.cc
            )
endif()
### runtime framework
file(GLOB_RECURSE OPS_SRC ${LITE_DIR}/src/ops/*.cc)
set(TEST_LITE_SRC
        ${ANF_SRC}
        ${OPS_SRC}
        ${KERNEL_OP_SRC}
        ${LITE_DIR}/src/runtime/allocator.cc
        ${LITE_DIR}/src/runtime/runtime_api.cc
        ${LITE_DIR}/src/runtime/thread_pool.cc
        ${LITE_DIR}/src/runtime/workspace_pool.cc
        ${LITE_DIR}/src/ir/tensor.cc
        ${LITE_DIR}/src/context.cc
        ${LITE_DIR}/src/executor.cc
        ${LITE_DIR}/src/kernel_factory.cc
        ${LITE_DIR}/src/kernel_registry.cc
        ${LITE_DIR}/src/lite_kernel.cc
        ${LITE_DIR}/src/lite_session.cc
        ${LITE_DIR}/src/model.cc
        ${LITE_DIR}/src/model_impl.cc
        ${LITE_DIR}/src/populate_parameter.cc
        ${LITE_DIR}/src/scheduler.cc
        ${LITE_DIR}/src/common/graph_util.cc
        ${LITE_DIR}/src/common/file_utils.cc
        ${LITE_DIR}/src/common/file_utils_ext.cc
        ${LITE_DIR}/src/common/utils.cc
        ${LITE_DIR}/src/common/ms_tensor_utils.cc
        ${LITE_DIR}/tools/common/graph_util.cc
        ${LITE_DIR}/tools/common/tensor_util.cc
        ${LITE_DIR}/tools/common/node_util.cc
        ${LITE_DIR}/tools/common/flag_parser.cc
        ${LITE_DIR}/tools/common/storage.cc
        ${LITE_DIR}/tools/benchmark/benchmark.cc
        ${LITE_DIR}/test/st/benchmark_test.cc
        )
### gpu runtime
if (SUPPORT_GPU)
    include_directories(${TOP_DIR}/third_party/OpenCL-Headers)
    include_directories(${TOP_DIR}/third_party/OpenCL-CLHPP/include)
    set(OPENCL_RUNTIME_SRC
            ${LITE_DIR}/src/runtime/opencl/opencl_allocator.cc
            ${LITE_DIR}/src/runtime/opencl/opencl_executor.cc
            ${LITE_DIR}/src/runtime/opencl/opencl_runtime.cc
            ${LITE_DIR}/src/runtime/opencl/opencl_wrapper.cc
            )
    set(TEST_LITE_SRC
            ${TEST_LITE_SRC}
            ${OPENCL_RUNTIME_SRC}
            )
endif()
### converter
if(BUILD_CONVERTER)
    file(GLOB_RECURSE TEST_CASE_TFLITE_PARSERS_SRC
            ${TEST_DIR}/ut/tools/converter/parser/tflite/*.cc
            )
    set(TEST_LITE_SRC
            ${TEST_LITE_SRC}
            ${TEST_CASE_TFLITE_PARSERS_SRC}
            ${TOP_DIR}/mindspore/core/utils/flags.cc
            ${LITE_DIR}/tools/converter/optimizer.cc
            # ${LITE_DIR}/src/common/anf_importer/import_from_protobuf.cc
            # ${LITE_DIR}/src/common/anf_importer/import_from_meta_graphT.cc
            # ${LITE_DIR}/src/common/anf_importer/import_from_protobuf.cc
            ${LITE_DIR}/tools/converter/anf_transform.cc
            ${LITE_DIR}/tools/converter/graphdef_transform.cc
            ${LITE_DIR}/tools/converter/converter_flags.cc
            ${LITE_DIR}/tools/converter/converter.cc
            ${LITE_DIR}/test/st/converter_test.cc
            ${LITE_DIR}/test/ut/tools/optimizer/fusion/conv_activation_fusion_test.cc
            ${LITE_DIR}/test/ut/tools/optimizer/fusion/conv_biasadd_fusion_test.cc
            ${LITE_DIR}/test/ut/tools/optimizer/fusion/conv_bn_fusion_test.cc
            ${LITE_DIR}/test/ut/tools/optimizer/fusion/conv_scale_fusion_test.cc
            ${LITE_DIR}/test/ut/tools/optimizer/fusion/constant_folding_fusion_test.cc
            ${LITE_DIR}/tools/optimizer/common/node_pass_extends.cc
            ${LITE_DIR}/tools/optimizer/common/pass_manager_extends.cc
            ${LITE_DIR}/tools/optimizer/common/gllo_utils.cc
            ${LITE_DIR}/tools/optimizer/fusion/conv_biasadd_fusion.cc
            ${LITE_DIR}/tools/optimizer/fusion/conv_activation_fusion.cc
            ${LITE_DIR}/tools/optimizer/fusion/conv_transform_fusion.cc
            ${LITE_DIR}/tools/optimizer/fusion/conv_scale_fusion.cc
            ${LITE_DIR}/tools/optimizer/fusion/conv_bn_fusion.cc
            ${LITE_DIR}/tools/optimizer/fusion/constant_folding_fusion.cc
            )
endif()
### train
if (SUPPORT_TRAIN)
    set(TEST_LITE_SRC
            ${TEST_LITE_SRC}
            #            ${SRC_DIR}/common/trans.cc
            #            ${SRC_DIR}/common/lite/trans_extends.cc
            #            ${SRC_DIR}/kernel/kernel_build_info.cc
            #            ${SRC_DIR}/utils/lite/base_ref_utils.cc
            #            ${SRC_DIR}/session/lite/anf_runtime_algorithm_extends.cc
            #            ${SRC_DIR}/session/lite/session_basic_extends.cc
            #            ${SRC_DIR}/session/anf_runtime_algorithm.cc
            #            ${SRC_DIR}/session/anf_runtime_algorithm.cc
            #            ${SRC_DIR}/session/session_basic.cc
            #            ${SRC_DIR}/session/kernel_graph.cc
            #            ${SRC_DIR}/session/session_factory.cc
            #            ${SRC_DIR}/device/kernel_info.cc
            #            ${SRC_DIR}/device/kernel_runtime.cc
            #            ${SRC_DIR}/device/lite/kernel_runtime_extends.cc
    #        ${LITE_DIR}/src/common/anf_importer/anf_importer.cc
    #       ${LITE_DIR}/src/common/anf_importer/import_from_meta_graph.cc
    #       ${LITE_DIR}/src/ir/primitive_value.cc
    #       ${LITE_DIR}/src/train/lite_kernel_runtime.cc
    #       ${LITE_DIR}/src/train/train_session.cc
    #       ${LITE_DIR}/src/train/model_impl.cc
            ${LITE_DIR}/src/lite_session.cc  # temporary
            )
else()
    set(TEST_LITE_SRC
            ${TEST_LITE_SRC}
            ${LITE_DIR}/src/lite_session.cc
            )
endif()
### test src
file(GLOB_RECURSE TEST_CASE_KERNEL_SRC
    ${TEST_DIR}/ut/src/runtime/kernel/arm/common/*.cc
    ${TEST_DIR}/ut/src/runtime/kernel/arm/fp32/*.cc
    ${TEST_DIR}/ut/src/runtime/kernel/arm/int8/*.cc
)

file(GLOB_RECURSE TEST_CASE_KERNEL_TRAIN_SRC
    ${TEST_DIR}/ut/src/runtime/kernel/arm/fp32_grad/*.cc
)

set(TEST_SRC
    ${TEST_LITE_SRC}
    ${TEST_MINDDATA_SRC}
    ${TEST_CASE_KERNEL_SRC}
    ${TEST_DIR}/common/common_test.cc
    ${TEST_DIR}/main.cc
    ${TEST_DIR}/ut/src/runtime/kernel/arm/common/pack_tests.cc
    ${TEST_DIR}/ut/src/infer_test.cc
    ${TEST_DIR}/ut/src/utils_test.cc
)

if (SUPPORT_TRAIN)
    set(TEST_SRC
            ${TEST_SRC}
            ${TEST_CASE_KERNEL_TRAIN_SRC}
  #         ${TEST_DIR}/ut/src/train_test.cc
            ${TEST_DIR}/ut/src/infer_test.cc  # temporary
            )
else()
    set(TEST_SRC
            ${TEST_SRC}
            ${TEST_DIR}/ut/src/infer_test.cc
            )
endif()

if (SUPPORT_GPU)
    set(TEST_SRC
            ${TEST_SRC}
            ${TEST_DIR}/ut/src/runtime/kernel/opencl/matmul_tests.cc
            ${TEST_DIR}/ut/src/runtime/kernel/opencl/depthwise_conv2d_tests.cc
            ${TEST_DIR}/ut/src/runtime/kernel/opencl/concat_tests.cc
            ${TEST_DIR}/ut/src/runtime/kernel/opencl/softmax_tests.cc
            ${TEST_DIR}/ut/src/runtime/kernel/opencl/arithmetic_tests.cc
            ${TEST_DIR}/ut/src/runtime/kernel/opencl/avg_pooling_tests.cc
            ${TEST_DIR}/ut/src/runtime/kernel/opencl/max_pooling_tests.cc
            ${TEST_DIR}/ut/src/runtime/kernel/opencl/utils_tests.cc
            ${TEST_DIR}/ut/src/runtime/kernel/opencl/conv2d_transpose_tests.cc
            ${TEST_DIR}/ut/src/runtime/kernel/opencl/transpose_tests.cc
            ${TEST_DIR}/ut/src/runtime/kernel/opencl/convolution_tests.cc
            ${TEST_DIR}/ut/src/runtime/kernel/opencl/activation_tests.cc
            )
endif()

if (ENABLE_FP16)
    set(TEST_SRC
            ${TEST_SRC}
)
endif ()


add_executable(lite-test ${TEST_SRC})

target_link_libraries(lite-test dl ${GTEST_LIBRARY})
if (BUILD_MINDDATA)
    target_link_libraries(lite-test
           minddata-lite
           minddata-eager
           )
    if (PLATFORM_ARM32 OR PLATFORM_ARM64)
           target_link_libraries(lite-test log)
    endif()
endif()
if (BUILD_CONVERTER)
    target_link_libraries(lite-test
            anf_importer_mid
            tflite_parser_mid
            caffe_parser_mid
            onnx_parser_mid
            node_mid
            graph_pass_mid
            fusion_mid
            quantizer_mid
            pthread
            protobuf
            mindspore::eigen
            mindspore::json
            ${SECUREC_LIBRARY}
            )
endif()
