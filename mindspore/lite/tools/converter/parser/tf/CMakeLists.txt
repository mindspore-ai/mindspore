include(${TOP_DIR}/mindspore/lite/cmake/merge.cmake)
merge_parser(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_BINARY_DIR}/tools/converter/parser/tf/tf_op_parser.cc)
file(GLOB_RECURSE TF_SRC_LIST ${CMAKE_BINARY_DIR}/tools/converter/parser/tf/tf_op_parser.cc)

set_property(SOURCE ${TF_SRC_LIST} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_LITE)

add_library(tf_parser_mid OBJECT ${TF_SRC_LIST})

add_dependencies(tf_parser_mid proto_mid)
add_dependencies(tf_parser_mid fbs_src)
add_dependencies(tf_parser_mid fbs_inner_src)
