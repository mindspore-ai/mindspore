file(GLOB_RECURSE _PIPELINE_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
    "pipeline.cc"
    "resource.cc"
    "pass.cc"
    "action.cc"
    "validator.cc"
    "remove_value_node_dup.cc"
    "parse/*.cc"
    "static_analysis/*.cc"
)

if (ENABLE_GE OR ENABLE_D)
    file(GLOB_RECURSE _PIPELINE_GE_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "pipeline_ge.cc")
    list(APPEND _PIPELINE_SRC_FILES ${_PIPELINE_GE_SRC_FILES})
endif ()

add_library(_mindspore_pipeline_obj OBJECT ${_PIPELINE_SRC_FILES})
