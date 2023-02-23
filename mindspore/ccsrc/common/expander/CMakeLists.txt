file(GLOB_RECURSE _EXPANDER_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        "*.cc"
        )

add_library(_mindspore_common_expander_obj OBJECT ${_EXPANDER_SRC_LIST})
