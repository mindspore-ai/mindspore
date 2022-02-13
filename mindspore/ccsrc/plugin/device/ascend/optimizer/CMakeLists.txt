if("${ENABLE_HIDDEN}" STREQUAL "OFF")
    string(REPLACE " -Werror " " " CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    string(REPLACE " -fvisibility=hidden" " -fvisibility=default" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endif()

if(ENABLE_D OR ENABLE_ACL)
    file(GLOB_RECURSE _PREACTIVATE_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        "*.cc"
    )
endif()

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -Wno-user-defined-warnings -Wno-inconsistent-missing-override \
        -Wno-overloaded-virtual -Wno-unused-const-variable -Wno-pessimizing-move")
endif()

set_property(SOURCE ${_PREACTIVATE_SRC_LIST} PROPERTY COMPILE_DEFINITIONS
  SUBMODULE_ID=mindspore::SubModuleId::SM_PRE_ACT)
if(ENABLE_D OR ENABLE_ACL)
    add_library(_mindspore_plugin_device_ascend_optimizer_obj OBJECT ${_PREACTIVATE_SRC_LIST})
endif()
