set(ANF_SRC
        ${ANF_SRC}
#core / abstract
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../core/abstract/abstract_function.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../core/abstract/analysis_context.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../core/abstract/param_validator.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../core/abstract/abstract_value.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../core/abstract/dshape.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../core/abstract/utils.cc
#core / base
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../core/base/base_ref.cc
#core / ir
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../core/ir/anf.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../core/ir/anf_extends.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../core/ir/meta_func_graph.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../core/ir/func_graph.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../core/ir/graph_utils.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../core/ir/func_graph_cloner.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../core/ir/func_graph_extends.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../core/ir/manager.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../core/ir/primitive.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../core/ir/tensor.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../core/ir/visitor.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../core/ir/meta_tensor_extends.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../core/ir/dtype.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../core/ir/dtype_extends.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../core/ir/named.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../core/ir/scope.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../core/ir/value.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../core/ir/value_extends.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../core/ir/dtype/container.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../core/ir/dtype/empty.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../core/ir/dtype/number.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../core/ir/dtype/ref.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../core/ir/dtype/type.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../core/ir/dtype/type_extends.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../core/utils/any.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../core/utils/symbolic.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../core/utils/misc.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../core/utils/flags.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../core/utils/trace_base.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../core/utils/trace_info.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../core/utils/label.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../core/utils/info.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../core/utils/profile.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../core/utils/ms_context.cc
        ## ccsrc
        ${CCSRC_DIR}/debug/draw.cc
        ${CCSRC_DIR}/pybind_api/export_flags.cc
        ${CCSRC_DIR}/utils/context/context_extends.cc
        ${CCSRC_DIR}/frontend/parallel/costmodel_context.cc
        ${CCSRC_DIR}/backend/optimizer/common/pattern_engine.cc
        ${CCSRC_DIR}/backend/optimizer/common/visit.cc
        ${CCSRC_DIR}/backend/optimizer/common/optimizer.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../../src/common/graph_utils_extends.cc
        )

file(GLOB_RECURSE OPS_SRC ${CMAKE_CURRENT_SOURCE_DIR}/../../src/ops/*.cc)

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
        ${CMAKE_CURRENT_SOURCE_DIR}/../common/flag_parser.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../common/storage.cc
        # ${CMAKE_CURRENT_SOURCE_DIR}/../../src/common/anf_exporter/anf_exporter.cc

        ../optimizer/common/node_pass_extends.cc
        ../optimizer/common/pass_manager_extends.cc
        ../optimizer/common/gllo_utils.cc
        ../optimizer/fusion/conv_biasadd_fusion.cc
        ../optimizer/fusion/conv_activation_fusion.cc
        ../optimizer/fusion/conv_transform_fusion.cc
        ../optimizer/fusion/conv_scale_fusion.cc
        ../optimizer/fusion/conv_bn_fusion.cc
        ../optimizer/fusion/constant_folding_fusion.cc
        )

add_subdirectory(parser/caffe)
add_subdirectory(parser/tflite)
add_subdirectory(parser/onnx)
add_subdirectory(legacy_optimizer)
add_subdirectory(quantizer)

add_executable(converter_lite
        main.cc
        ${ANF_SRC}
        ${CONVERTER_SRC}
        ${OPS_SRC}
        )
target_link_libraries(converter_lite PRIVATE
        tflite_parser_mid
        caffe_parser_mid
        onnx_parser_mid
        anf_importer_mid
        node_mid
        graph_pass_mid
        fusion_mid
        quantizer_mid
        protobuf
        quantizer_mid
        pthread
        mindspore-lite
        ${SECUREC_LIBRARY}
        mindspore::json
        mindspore::eigen
        )

