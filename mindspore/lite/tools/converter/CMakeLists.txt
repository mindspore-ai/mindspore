add_definitions(-DPRIMITIVE_WRITEABLE)
add_definitions(-DUSE_GLOG)

set(CCSRC_SRC
        ${CCSRC_DIR}/backend/optimizer/common/pattern_engine.cc
        ${CCSRC_DIR}/backend/optimizer/common/visit.cc
        ${CCSRC_DIR}/backend/optimizer/common/optimizer.cc
        )

include(${TOP_DIR}/cmake/external_libs/glog.cmake)

file(GLOB OPS_SRC ${CMAKE_CURRENT_SOURCE_DIR}/../../src/ops/*.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../../src/ops/populate/*.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/ops/*.cc
        )

file(GLOB_RECURSE CONVERTER_SRC RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../flag/flag_parser.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/converter.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/converter_flags.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/anf_transform.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/graphdef_transform.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/optimizer.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../../src/common/file_utils.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../common/graph_util.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../common/node_util.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../common/tensor_util.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../common/string_util.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../common/protobuf_utils.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../common/flag_parser.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../common/storage.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../../src/ir/primitive_t_value.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../../src/common/utils.cc

        ../optimizer/common/node_pass_extends.cc
        ../optimizer/common/pass_manager_extends.cc
        ../optimizer/common/gllo_utils.cc
        ../optimizer/fusion/conv_biasadd_fusion.cc
        ../optimizer/fusion/conv_activation_fusion.cc
        ../optimizer/fusion/conv_tuple_activation_fusion.cc
        ../optimizer/fusion/conv_transform_fusion.cc
        ../optimizer/fusion/conv_scale_fusion.cc
        ../optimizer/fusion/conv_bn_fusion.cc
        ../optimizer/fusion/conv_tuplegetitem_fusion.cc
        ../optimizer/fusion/constant_folding_fusion.cc
        ../optimizer/fusion/quant_dtype_cast_fusion.cc
        ../optimizer/fusion/tf_norm_fusion.cc
        ../optimizer/fusion/onnx_layer_norm_fusion.cc
        ../optimizer/fusion/batchmatmul_fusion.cc
        ../optimizer/fusion/sigmoid_mul_fusion.cc
        ../optimizer/fusion/conv_conv_fusion.cc
        ../optimizer/fusion/tflite_lstm_cell_fusion.cc
        ../optimizer/fusion/tf_lstm_cell_fusion.cc
        ../optimizer/fusion/tf_bidirection_gru_fusion.cc
        ../optimizer/fusion/tf_bidirection_gru_cf_fusion.cc
        ../optimizer/graph/weight_format_transform_pass.cc
        ../optimizer/graph/weight_format_hardcode_pass.cc
        ../optimizer/graph/clip_convert_activation_pass.cc
        ../optimizer/graph/group_depthwise_op_convert_pass.cc
        ../optimizer/graph/tflite_inputs_adjust_pass.cc
        ../optimizer/graph/update_conv2d_param_pass.cc
        ../optimizer/graph/unused_node_remove_pass.cc
        ../optimizer/graph/unused_cast_node_remove_pass.cc
        ../optimizer/graph/unused_transpose_node_remove_pass.cc
        ../optimizer/graph/redundant_op_remove_pass.cc
        ../optimizer/graph/infershape_pass.cc
        ../optimizer/graph/slice_prepose_pass.cc
        ../optimizer/graph/mindir_adjust_pass.cc
        ../optimizer/graph/onnx_inputs_adjust_pass.cc
        ../optimizer/graph/onnx_pad_adjust_pass.cc
        ../optimizer/graph/while_pass.cc
        ../optimizer/graph/if_pass.cc
        ../optimizer/graph/functionalize_control_op_pass.cc
        ../optimizer/graph/functionalize_while.cc
        ../optimizer/graph/functionalize_cond.cc
        ../optimizer/graph/inputs_adjust_pass.cc
        ../optimizer/graph/primitive_adjust_pass.cc
        )

add_subdirectory(../anf_exporter anf_exporter)
add_subdirectory(parser/caffe)
add_subdirectory(parser/tflite)
add_subdirectory(parser/onnx)
add_subdirectory(parser/tf)
add_subdirectory(legacy_optimizer)
add_subdirectory(quantizer)
add_subdirectory(${CORE_DIR} mindspore_core)

set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../src)
set(LITE_SRC
        ${SRC_DIR}/common/graph_util.cc
        ${SRC_DIR}/common/string_util.cc
        ${SRC_DIR}/common/prim_util.cc
        ${SRC_DIR}/common/tensor_util.cc
        ${SRC_DIR}/runtime/allocator.cc
        ${SRC_DIR}/runtime/runtime_api.cc
        ${SRC_DIR}/runtime/thread_pool.c
        ${SRC_DIR}/runtime/infer_manager.cc
        ${SRC_DIR}/inner_context.cc
        ${SRC_DIR}/tensor.cc
        ${SRC_DIR}/tensorlist.cc
        ${SRC_DIR}/kernel_registry.cc
        ${SRC_DIR}/lite_kernel.cc
        ${SRC_DIR}/scheduler.cc
        ${SRC_DIR}/sub_graph_kernel.cc
        ${SRC_DIR}/lite_session.cc
        ${SRC_DIR}/executor.cc
        ${SRC_DIR}/lite_model.cc
        ${SRC_DIR}/errorcode.cc
        ${SRC_DIR}/dequant.cc
        ${SRC_DIR}/huffman_decode.cc
        ${SRC_DIR}/ops/ops_utils.cc
        ${SRC_DIR}/ops/ops_def.cc
        ${SRC_DIR}/train/train_populate_parameter.cc
        )

set(ENABLE_MINDRT "off")
if(ENABLE_MINDRT)
    include_directories(${CORE_DIR}/mindrt/)
    include_directories(${CORE_DIR}/mindrt/src/)
    set(MINDRT_SRC
        ${SRC_DIR}/lite_mindrt.cc
        ${SRC_DIR}/mindrt_executor.cc
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

set(ARM_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../src/runtime/kernel/arm)
file(GLOB KERNEL_SRC
        ${ARM_DIR}/base/*.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../../nnacl/*.c
        ${CMAKE_CURRENT_SOURCE_DIR}/../../nnacl/fp32/*.c
        ${CMAKE_CURRENT_SOURCE_DIR}/../../nnacl/infer/*.c
        ${CMAKE_CURRENT_SOURCE_DIR}/../../nnacl/int8/*.c
        ${CMAKE_CURRENT_SOURCE_DIR}/../../nnacl/base/*.c
        ${CMAKE_CURRENT_SOURCE_DIR}/../../nnacl/quantization/*.c
        ${ARM_DIR}/fp32/*.cc
        ${ARM_DIR}/int8/*.cc
        )

if(PLATFORM_ARM64)
    # assembly
    file(GLOB ASSEMBLY_SRC ${CMAKE_CURRENT_SOURCE_DIR}/../../nnacl/assembly/arm64/*.s
            ${CMAKE_CURRENT_SOURCE_DIR}/../../nnacl/assembly/arm64/*.S)
    set_property(SOURCE ${ASSEMBLY_SRC} PROPERTY LANGUAGE C)
    set(KERNEL_SRC ${KERNEL_SRC} ${ASSEMBLY_SRC})
endif()

file(GLOB PROTO_FILE ""
        ${CMAKE_CURRENT_SOURCE_DIR}/parser/caffe/caffe.proto
        ${CMAKE_CURRENT_SOURCE_DIR}/parser/tf/proto/*.proto
        ${CMAKE_CURRENT_SOURCE_DIR}/parser/onnx/onnx.proto
        ${CCSRC_DIR}/utils/mind_ir.proto)

ms_protobuf_generate(PROTO_SRCS PROTO_HDRS ${PROTO_FILE})
add_library(proto_mid OBJECT ${PROTO_SRCS})
set(TFLITE_FBS_FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/parser/tflite/schema.fbs
        )
ms_build_flatbuffers_lite(TFLITE_FBS_FILES ${CMAKE_CURRENT_SOURCE_DIR}/parser/tflite/ tflite_fbs_src
        ${CMAKE_BINARY_DIR}/schema "inner")

set_property(SOURCE ${CONVERTER_SRC} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_LITE)
set_property(SOURCE ${CCSRC_SRC} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_LITE)
set_property(SOURCE ${OPS_SRC} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_LITE)
set_property(SOURCE ${KERNEL_SRC} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_LITE)
set_property(SOURCE ${LITE_SRC} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_LITE)
set_property(SOURCE ${MINDRT_SRC} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_LITE)
add_executable(converter_lite
        main.cc
        ${CCSRC_SRC}
        ${CONVERTER_SRC}
        ${OPS_SRC}
        ${KERNEL_SRC}
        ${LITE_SRC}
        ${MINDRT_SRC}
        )
add_dependencies(converter_lite fbs_src)
add_dependencies(converter_lite fbs_inner_src)

target_link_libraries(converter_lite PRIVATE
        tflite_parser_mid
        tf_parser_mid
        caffe_parser_mid
        onnx_parser_mid
        anf_exporter_mid
        graph_pass_mid
        fusion_mid
        quantizer_mid
        proto_mid
        ${SECUREC_LIBRARY}
        mindspore::json
        mindspore::eigen
        -Wl,--whole-archive mindspore_core -Wl,--no-whole-archive
        mindspore::glog
        mindspore::protobuf
        mindspore::flatbuffers
        pthread
        )
