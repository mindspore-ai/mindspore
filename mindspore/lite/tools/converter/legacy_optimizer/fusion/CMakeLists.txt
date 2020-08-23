add_library(fusion_mid OBJECT
        ${CMAKE_CURRENT_SOURCE_DIR}/fusion_pattern.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/fusion_pass.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/matmul_biasadd_fusion_pass.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/quant_cast_fusion_pass.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/batchnorm_fold_fusion_pass.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/format_trans_fusion_pass.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/format_trans_transpose_fusion_pass.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/batchnorm_convert_scale_pass.cc
        )

target_link_libraries(fusion_mid securec)
