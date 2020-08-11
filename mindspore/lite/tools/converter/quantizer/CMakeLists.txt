set(3RD_DIR ../../../third_party)
include_directories(${3RD_DIR}/protobuf/build/include)
include_directories(${3RD_DIR}/flatbuffers/include)
include_directories(${3RD_DIR}/opencv/build/include/opencv4)

add_library(quantizer_mid OBJECT
    ${CMAKE_CURRENT_SOURCE_DIR}/calc_quant_param.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/quantizer.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/aware_quantizer.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/weight_quantizer.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/quantize_util.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/general_bitpacking.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/post_training_quantizer.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/quant_cast.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../src/common/anf_exporter/anf_exporter.cc
    )

if(ENABLE_ASAN)
  target_link_libraries(quantizer_mid libasan libSecodefuzz)
endif()
