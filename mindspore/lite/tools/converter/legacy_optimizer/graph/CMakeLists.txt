add_library(graph_pass_mid OBJECT
        ${CMAKE_CURRENT_SOURCE_DIR}/format_trans_pass.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/trans_format_insert_pass.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/dtype_trans_pass.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/isolated_node_remove_pass.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/model_input_format_preprocess_pass.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/weight_format_hardcode_pass.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/weight_format_transform_pass.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/topological_sort_pass.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/unused_node_remove_pass.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/batchnorm_convert_scale_pass.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/trans_format_remove_pass.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/infershape_pass.cc
        )
