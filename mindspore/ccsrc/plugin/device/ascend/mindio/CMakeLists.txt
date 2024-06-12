if(NOT ENABLE_SECURITY AND ENABLE_D)
    list(APPEND MINDIO_SRC_LIST ${CMAKE_CURRENT_SOURCE_DIR}/mindio_adapter.cc)

    set_property(SOURCE ${MINDIO_SRC_LIST} PROPERTY COMPILE_DEFINITIONS
            SUBMODULE_ID=mindspore::SubModuleId::SM_MINDIO)
    add_library(_mindspore_plugin_device_ascend_mindio_obj OBJECT ${MINDIO_SRC_LIST})
endif()