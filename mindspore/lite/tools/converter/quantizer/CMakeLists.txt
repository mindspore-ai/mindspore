set(3RD_DIR ../../../third_party)
include_directories(${3RD_DIR}/protobuf/build/include)
include_directories(${3RD_DIR}/flatbuffers/include)
include_directories(${3RD_DIR}/opencv/build/include/opencv4)

file(GLOB QUANTIZER
        ${CMAKE_CURRENT_SOURCE_DIR}/quant_helper/*
        ${CMAKE_CURRENT_SOURCE_DIR}/quantizer.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/quantize_util.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/full_quant_quantizer.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/quant_cast.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/weight_quantizer.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/huffman_encode.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/fse_decoder.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/fse_bit_stream.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/fse_encoder.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/mixed_bit_weight_quantizer.cc
        )
set_property(SOURCE ${QUANTIZER} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_LITE)
add_library(quantizer_mid OBJECT ${QUANTIZER})

add_dependencies(quantizer_mid fbs_src)
add_dependencies(quantizer_mid fbs_inner_src)
