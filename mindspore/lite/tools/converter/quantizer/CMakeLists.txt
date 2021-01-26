set(3RD_DIR ../../../third_party)
include_directories(${3RD_DIR}/protobuf/build/include)
include_directories(${3RD_DIR}/flatbuffers/include)
include_directories(${3RD_DIR}/opencv/build/include/opencv4)

file(GLOB QUANTIZER
        ${CMAKE_CURRENT_SOURCE_DIR}/calc_quant_param.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/quantizer.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/quantize_util.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/post_training_quantizer.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/quant_cast.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/weight_quantizer.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/huffman_encode.cc
        )
set_property(SOURCE ${QUANTIZER} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_LITE)
add_library(quantizer_mid OBJECT ${QUANTIZER})

add_dependencies(quantizer_mid fbs_src)
add_dependencies(quantizer_mid fbs_inner_src)
if(ENABLE_ASAN)
    target_link_libraries(quantizer_mid libasan libSecodefuzz)
endif()
