file(GLOB_RECURSE _OPERATOR_ALL_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        "*.cc"
        )

add_library(_mindspore_operator_obj OBJECT ${_OPERATOR_ALL_SRC_FILES})
