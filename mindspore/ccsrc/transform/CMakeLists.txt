file(GLOB_RECURSE _TRANSFORM_ALL_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        "*.cc"
        )

add_library(_mindspore_transform_obj OBJECT ${_TRANSFORM_ALL_SRC_FILES})
