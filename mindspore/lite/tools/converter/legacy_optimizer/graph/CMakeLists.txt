file(GLOB GRAPH_PASS
        ${CMAKE_CURRENT_SOURCE_DIR}/format_trans_pass.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/trans_format_insert_pass.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/dtype_trans_pass.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/isolated_node_remove_pass.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/model_input_format_preprocess_pass.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/topological_sort_pass.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/dropout_node_remove_pass.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/batchnorm_convert_scale_pass.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/trans_format_remove_pass.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/infershape_pass.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/tensor_quant_pass.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/infer_quant_param_pass.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/global_format_transform_pass.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/set_unused_quant_param_to_default_pass.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/tensor_name_pass.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/switch_pass.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/select_pass.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/subgraph_node_pass.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/subgraph_tensor_pass.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/nested_loop_expand_pass.cc
        )
set_property(SOURCE ${GRAPH_PASS} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_LITE)
add_library(graph_pass_mid OBJECT ${GRAPH_PASS})

add_dependencies(graph_pass_mid fbs_src)
add_dependencies(graph_pass_mid fbs_inner_src)
