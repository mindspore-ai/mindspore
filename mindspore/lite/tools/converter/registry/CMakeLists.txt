set(KERNEL_REG_DIR ${TOP_DIR}/mindspore/lite/src/registry)
file(GLOB CONVERT_REG_SRC
        pass_registry.cc
        model_parser_registry.cc
        )
file(GLOB KERNEL_REG_SRC ${KERNEL_REG_DIR}/*.cc)
set(REG_SRC ${CONVERT_REG_SRC}
        ${KERNEL_REG_SRC}
        ${CORE_DIR}/utils/log_adapter.cc
        ${CORE_DIR}/gvar/log_adapter_common.cc
        ${CORE_DIR}/gvar/logging_level.cc)
set_property(SOURCE ${REG_SRC} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_LITE)
add_library(mslite_converter_plugin_reg SHARED ${REG_SRC})
target_link_libraries(mslite_converter_plugin_reg mindspore::glog)
add_dependencies(mslite_converter_plugin_reg fbs_src)
add_dependencies(mslite_converter_plugin_reg fbs_inner_src)
