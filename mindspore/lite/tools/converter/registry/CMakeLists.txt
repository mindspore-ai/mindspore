file(GLOB CONVERT_REG_SRC
        pass_registry.cc
        model_parser_registry.cc
        )

set_property(SOURCE ${CONVERT_REG_SRC} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_LITE)
add_library(mslite_converter_plugin_reg SHARED ${CONVERT_REG_SRC})

add_dependencies(mslite_converter_plugin_reg fbs_src)
add_dependencies(mslite_converter_plugin_reg fbs_inner_src)
