if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-delete-abstract-non-virtual-dtor")
endif()

file(GLOB_RECURSE MS_DEVICE_CONTEXT_CONF RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        "op_precision_conf.cc"
        "op_tuning_conf.cc"
        "op_debug_conf.cc"
)

set_property(SOURCE ${MS_DEVICE_CONTEXT_CONF} PROPERTY COMPILE_DEFINITIONS
        SUBMODULE_ID=mindspore::SubModuleId::SM_DEVICE)

add_library(_mindspore_ascend_context_conf_obj OBJECT ${MS_DEVICE_CONTEXT_CONF})