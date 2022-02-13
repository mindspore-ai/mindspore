file(GLOB_RECURSE HARDWARE_ASCEND_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-delete-abstract-non-virtual-dtor")
endif()

set_property(SOURCE ${HARDWARE_ASCEND_SRC_LIST} PROPERTY COMPILE_DEFINITIONS
        SUBMODULE_ID=mindspore::SubModuleId::SM_DEVICE)
if(ENABLE_D)
    add_library(_mindspore_plugin_device_ascend_hal_hardware_obj OBJECT ${HARDWARE_ASCEND_SRC_LIST})
endif()
