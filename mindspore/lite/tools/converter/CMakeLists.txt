add_definitions(-DPRIMITIVE_WRITEABLE)
add_definitions(-DUSE_GLOG)

set(CCSRC_SRC
        ${CCSRC_DIR}/backend/optimizer/common/pattern_engine.cc
        ${CCSRC_DIR}/backend/optimizer/common/visit.cc
        ${CCSRC_DIR}/backend/optimizer/common/optimizer.cc
        )

include(${TOP_DIR}/cmake/external_libs/glog.cmake)
include_directories(${TOP_DIR}/mindspore/ccsrc/backend/kernel_compiler/cpu)

file(GLOB_RECURSE CONVERTER_SRC RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/ops/*.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../flag/flag_parser.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/converter.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/converter_flags.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/anf_transform.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/graphdef_transform.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/optimizer.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/export_model.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../../src/common/file_utils.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../../src/common/quant_utils.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../common/graph_util.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../common/meta_graph_utils.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../common/node_util.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../common/tensor_util.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../common/string_util.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../common/protobuf_utils.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../common/func_graph_subgraph.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../common/flag_parser.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../common/storage.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../../src/ir/primitive_t_value.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../../src/common/utils.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../../src/common/dynamic_library_loader.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/parser/parser_utils.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/parser/tf_bidirection_gru_cf_fusion.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/parser/unused_node_remove_pass.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/parser/conv1d_inout_adjust.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/parser/inputs_adjust.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/import/mindspore_importer.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/import/primitive_adjust.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/import/mindir_adjust.cc

        ../optimizer/common/node_pass_extends.cc
        ../optimizer/common/pass_manager_extends.cc
        ../optimizer/common/gllo_utils.cc
        ../optimizer/common/format_utils.cc
        ../optimizer/common/multiple_pattern_process_pass.cc
        ../optimizer/fusion/affine_activation_fusion.cc
        ../optimizer/fusion/affine_fusion.cc
        ../optimizer/fusion/conv_biasadd_fusion.cc
        ../optimizer/fusion/conv_activation_fusion.cc
        ../optimizer/fusion/conv_tuple_activation_fusion.cc
        ../optimizer/fusion/conv_transform_fusion.cc
        ../optimizer/fusion/conv_scale_fusion.cc
        ../optimizer/fusion/conv_bn_fusion.cc
        ../optimizer/fusion/conv_tuplegetitem_fusion.cc
        ../optimizer/fusion/constant_folding_fusion.cc
        ../optimizer/fusion/norm_fusion.cc
        ../optimizer/fusion/batchmatmul_fusion.cc
        ../optimizer/fusion/sigmoid_mul_fusion.cc
        ../optimizer/fusion/conv_conv_fusion.cc
        ../optimizer/fusion/tflite_lstm_cell_fusion.cc
        ../optimizer/fusion/tf_lstm_cell_fusion.cc
        ../optimizer/fusion/tf_bidirection_gru_fusion.cc
        ../optimizer/fusion/multi_head_attention_fusion.cc
        ../optimizer/fusion/reshape_reshape_fusion.cc
        ../optimizer/fusion/tflite_rel_pos_multi_head_attention_fusion.cc
        ../optimizer/fusion/glu_fusion.cc
        ../optimizer/fusion/matmul_add_fusion.cc
        ../optimizer/fusion/mul_add_fusion.cc
        ../optimizer/fusion/gelu_fusion.cc
        ../optimizer/fusion/tf_gelu_fusion.cc
        ../optimizer/fusion/onnx_gelu_fusion.cc
        ../optimizer/fusion/squeeze_fusion.cc
        ../optimizer/fisson/eliminate_concat_split.cc
        ../optimizer/fisson/fisson_util.cc
        ../optimizer/fisson/iter_node_outputs.cc
        ../optimizer/fisson/node_out_shapes.cc
        ../optimizer/fisson/multi_conv_split_pass.cc
        ../optimizer/parallel/multi_node_split.cc
        ../optimizer/parallel/multi_conv_info.cc
        ../optimizer/parallel/parallel_pass.cc
        ../optimizer/graph/add_tensor_array.cc
        ../optimizer/parallel/conv2d_info.cc
        ../optimizer/parallel/depthwise_conv2d_info.cc
        ../optimizer/parallel/operator_info.cc
        ../optimizer/parallel/operator_info_register.cc
        ../optimizer/parallel/spliter.cc
        ../optimizer/parallel/split_strategy.cc
        ../optimizer/graph/conv1d_weight_expanding_pass.cc
        ../optimizer/graph/clip_convert_activation_pass.cc
        ../optimizer/graph/group_depthwise_op_convert_pass.cc
        ../optimizer/graph/update_conv2d_param_pass.cc
        ../optimizer/graph/unused_cast_node_remove_pass.cc
        ../optimizer/graph/unused_transpose_node_remove_pass.cc
        ../optimizer/graph/redundant_op_remove_pass.cc
        ../optimizer/graph/infershape_pass.cc
        ../optimizer/graph/slice_prepose_pass.cc
        ../optimizer/graph/mindir_adjust_pass.cc
        ../optimizer/graph/control_flow_pass.cc
        ../optimizer/graph/primitive_adjust_pass.cc
        ../optimizer/graph/unify_format_pass.cc
        ../optimizer/graph/node_infershape.cc
        ../optimizer/graph/transpose_strategy.cc
        ../optimizer/graph/reduce_same_act_pass.cc
        ../optimizer/graph/split_one_pass.cc
        ../optimizer/graph/find_const_subgraph_pass.cc
        )

add_subdirectory(../anf_exporter anf_exporter)
add_subdirectory(parser/caffe)
add_subdirectory(parser/tflite)
add_subdirectory(parser/onnx)
add_subdirectory(parser/tf)
add_subdirectory(legacy_optimizer)
add_subdirectory(quantizer)
add_subdirectory(registry)
add_subdirectory(${CORE_DIR} mindspore_core)

set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../src)
set(API_SRC ${SRC_DIR}/cxx_api/context.cc)
set(LITE_SRC
        ${API_SRC}
        ${SRC_DIR}/common/context_util.cc
        ${SRC_DIR}/common/graph_util.cc
        ${SRC_DIR}/common/string_util.cc
        ${SRC_DIR}/common/prim_util.cc
        ${SRC_DIR}/common/tensor_util.cc
        ${SRC_DIR}/runtime/inner_allocator.cc
        ${SRC_DIR}/runtime/infer_manager.cc
        ${SRC_DIR}/inner_context.cc
        ${SRC_DIR}/tensor.cc
        ${SRC_DIR}/ms_tensor.cc
        ${SRC_DIR}/tensorlist.cc
        ${SRC_DIR}/registry/kernel_interface_registry.cc
        ${SRC_DIR}/registry/kernel_interface.cc
        ${SRC_DIR}/kernel_registry.cc
        ${SRC_DIR}/inner_kernel.cc
        ${SRC_DIR}/lite_kernel.cc
        ${SRC_DIR}/lite_kernel_util.cc
        ${SRC_DIR}/scheduler.cc
        ${SRC_DIR}/sub_graph_kernel.cc
        ${SRC_DIR}/sub_graph_split.cc
        ${SRC_DIR}/lite_session.cc
        ${SRC_DIR}/executor.cc
        ${SRC_DIR}/lite_model.cc
        ${SRC_DIR}/errorcode.cc
        ${SRC_DIR}/weight_decoder.cc
        ${SRC_DIR}/huffman_decode.cc
        ${SRC_DIR}/ops/ops_utils.cc
        ${SRC_DIR}/ops/ops_def.cc
        ${SRC_DIR}/train/train_populate_parameter.cc
        )

file(GLOB PROTO_FILE ""
        ${CMAKE_CURRENT_SOURCE_DIR}/parser/caffe/caffe.proto
        ${CMAKE_CURRENT_SOURCE_DIR}/parser/tf/proto/*.proto
        ${CMAKE_CURRENT_SOURCE_DIR}/parser/onnx/onnx.proto)

ms_protobuf_generate(PROTO_SRCS PROTO_HDRS ${PROTO_FILE})
add_library(proto_mid OBJECT ${PROTO_SRCS})
set(TFLITE_FBS_FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/parser/tflite/schema.fbs
        )
ms_build_flatbuffers_lite(TFLITE_FBS_FILES ${CMAKE_CURRENT_SOURCE_DIR}/parser/tflite/ tflite_fbs_src
        ${CMAKE_BINARY_DIR}/schema "inner")

set_property(SOURCE ${CONVERTER_SRC} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_LITE)
set_property(SOURCE ${CCSRC_SRC} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_LITE)
set_property(SOURCE ${LITE_SRC} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_LITE)

add_executable(converter_lite
        main.cc
        ${CCSRC_SRC}
        ${CONVERTER_SRC}
        ${LITE_SRC}
        )
add_dependencies(converter_lite fbs_src)
add_dependencies(converter_lite fbs_inner_src)

target_link_libraries(converter_lite PRIVATE
        mindrt_mid
        cpu_ops_mid
        nnacl_mid
        cpu_kernel_mid
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
        mslite_converter_plugin
        mindspore::glog
        mindspore::protobuf
        mindspore::flatbuffers
        )

if(NOT MSVC)
    target_link_libraries(converter_lite PRIVATE pthread)
endif()

if(NOT WIN32)
    target_link_libraries(converter_lite PRIVATE dl)
endif()

if(ENABLE_MODEL_OBF)
    target_link_libraries(converter_lite PRIVATE
            ${OBF_LIB_DIR}/libmsdeobfuscator-lite.so)
endif()
