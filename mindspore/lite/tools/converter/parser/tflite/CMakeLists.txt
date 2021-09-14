include(${TOP_DIR}/mindspore/lite/cmake/merge.cmake)
merge_parser(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_BINARY_DIR}/tools/converter/parser/tflite/tflite_op_parser.cc)
file(GLOB_RECURSE TFLITE_SRC_LIST ${CMAKE_BINARY_DIR}/tools/converter/parser/tflite/tflite_op_parser.cc)

set_property(SOURCE ${TFLITE_SRC_LIST} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_LITE)
add_library(tflite_parser_mid OBJECT
        ${TFLITE_SRC_LIST}
        )
add_dependencies(tflite_parser_mid tflite_fbs_src)
add_dependencies(tflite_parser_mid fbs_src)
add_dependencies(tflite_parser_mid fbs_inner_src)
target_link_libraries(tflite_parser_mid mindspore::flatbuffers)
