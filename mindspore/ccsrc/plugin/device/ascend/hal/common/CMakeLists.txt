file(GLOB_RECURSE COMMON_ASCEND_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")

set_property(SOURCE ${COMMON_ASCEND_SRC_LIST} PROPERTY COMPILE_DEFINITIONS
        SUBMODULE_ID=mindspore::SubModuleId::SM_COMMON)
if(ENABLE_D)
    add_library(_mindspore_plugin_device_ascend_hal_common_obj OBJECT ${COMMON_ASCEND_SRC_LIST})
endif()