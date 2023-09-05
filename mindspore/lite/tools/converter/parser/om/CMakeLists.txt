include(${TOP_DIR}/mindspore/lite/cmake/merge.cmake)
merge_parser(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_BINARY_DIR}/tools/converter/parser/om/om_op_parser.cc)
file(GLOB_RECURSE OM_SRC_LIST ${CMAKE_BINARY_DIR}/tools/converter/parser/om/om_op_parser.cc)
set_property(SOURCE ${OM_SRC_LIST} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_LITE)
add_library(om_parser_mid OBJECT
        ${OM_SRC_LIST}
        )
add_dependencies(om_parser_mid proto_mid)
add_dependencies(om_parser_mid fbs_src)
add_dependencies(om_parser_mid fbs_inner_src)
