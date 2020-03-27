file(GLOB_RECURSE _PIPELINE_ALL_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        "pipeline.cc"
        "resource.cc"
        "pass.cc"
        "action.cc"
        "validator.cc"
        "remove_value_node_dup.cc"
        "parse/*.cc"
        "static_analysis/*.cc"
        )

add_library(_mindspore_pipeline_obj OBJECT ${_PIPELINE_ALL_SRC_FILES})