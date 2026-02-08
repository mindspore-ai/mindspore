file(GLOB_RECURSE _PIPELINE_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
    "pipeline.cc"
    "pipeline_interface.cc"
    "executor/*.cc"
    "resource.cc"
    "resource_interface.cc"
    "pass_config.cc"
    "pass.cc"
    "action.cc"
    "validator.cc"
    "remove_value_node_dup.cc"
    "pipeline_split.cc"
    "compile_cache_manager.cc"
    "event_message_print.cc"
    "fallback.cc"
    "graph_circle_handler.cc"
    "parse/*.cc"
    "static_analysis/*.cc"
    "debug/*.cc"
    "load_mindir.cc"
)

file(GLOB PIPELINE_SRC_FILES "*.cc" "executor/*.cc")
set_property(SOURCE ${PIPELINE_SRC_FILES} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_PIPELINE)

file(GLOB_RECURSE PARSER_SRC_FILES "parse/*.cc")
set_property(SOURCE ${PARSER_SRC_FILES} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_PARSER)

file(GLOB_RECURSE ANALYZER_SRC_FILES "static_analysis/*.cc")
set_property(SOURCE ${ANALYZER_SRC_FILES} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_ANALYZER)

file(GLOB_RECURSE DEBUG_SRC_FILES "debug/*.cc")
set_property(SOURCE ${DEBUG_SRC_FILES} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_DEBUG)

if("${ENABLE_HIDDEN}" STREQUAL "OFF" AND NOT MSVC)
    string(REPLACE " -Werror " " " CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    string(REPLACE " -fvisibility=hidden" " -fvisibility=default" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endif()

add_library(_mindspore_frontend_jit_ps_obj OBJECT ${_PIPELINE_SRC_FILES})
