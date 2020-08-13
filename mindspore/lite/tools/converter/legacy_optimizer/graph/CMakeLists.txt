add_library(graph_pass_mid OBJECT
        ${CMAKE_CURRENT_SOURCE_DIR}/format_trans_pass.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/dtype_trans_pass.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/isolated_node_remove_pass.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/model_input_format_preprocess_pass.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/topological_sort_pass.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/unused_node_remove_pass.cc
        )
