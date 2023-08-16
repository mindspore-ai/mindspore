file(GLOB_RECURSE _BACKEND_EXPANDER_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        "*.cc"
        )
add_library(_mindspore_backend_common_expander_obj OBJECT ${_BACKEND_EXPANDER_SRC_LIST})
