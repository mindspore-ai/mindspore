file(GLOB_RECURSE _PIPELINE_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
    "pipeline.cc"
    "resource.cc"
    "pass.cc"
    "action.cc"
    "validator.cc"
    "remove_value_node_dup.cc"
    "pipeline_split.cc"
    "parse/*.cc"
    "static_analysis/*.cc"
)


file(GLOB PIPELINE_SRC_FILES "*.cc")
set_property(SOURCE ${PIPELINE_SRC_FILES} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_PIPELINE)

file(GLOB_RECURSE PARSER_SRC_FILES "parse/*.cc")
set_property(SOURCE ${PARSER_SRC_FILES} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_PARSER)

file(GLOB_RECURSE ANALYZER_SRC_FILES "static_analysis/*.cc")
set_property(SOURCE ${ANALYZER_SRC_FILES} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_ANALYZER)

if(ENABLE_GE OR ENABLE_D)
    file(GLOB_RECURSE _PIPELINE_GE_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "pipeline_ge.cc")
    list(APPEND _PIPELINE_SRC_FILES ${_PIPELINE_GE_SRC_FILES})
endif()

add_library(_mindspore_pipeline_jit_obj OBJECT ${_PIPELINE_SRC_FILES})
