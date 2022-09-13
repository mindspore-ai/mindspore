file(GLOB_RECURSE _SESSION_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
    "kernel_build_client.cc"
    "kernel_graph.cc"
    "kernel_graph_mgr.cc"
    "session_basic.cc"
    "session_factory.cc"
    "executor.cc"
    "executor_manager.cc"
    "anf_runtime_algorithm.cc"
    "debug_register.cc"
    "single_kernel_graph.cc"
)

if("${ENABLE_HIDDEN}" STREQUAL "OFF" AND NOT MSVC)
    string(REPLACE " -Werror " " " CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    string(REPLACE " -fvisibility=hidden" " -fvisibility=default" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endif()

if(CMAKE_SYSTEM_NAME MATCHES "Darwin")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-overloaded-virtual -Wno-delete-abstract-non-virtual-dtor")
endif()

set_property(SOURCE ${_SESSION_SRC_LIST} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_SESSION)
add_library(_mindspore_backend_common_session_obj OBJECT ${_SESSION_SRC_LIST})
