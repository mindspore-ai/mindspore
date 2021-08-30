file(GLOB_RECURSE CONFIG_PARSER_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        *.cc
        )
set_property(SOURCE ${CONFIG_PARSER_SRC_LIST} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_LITE)
add_library(config_parser_mid OBJECT
        ${CONFIG_PARSER_SRC_LIST}
        )
add_dependencies(config_parser_mid fbs_src)
add_dependencies(config_parser_mid fbs_inner_src)