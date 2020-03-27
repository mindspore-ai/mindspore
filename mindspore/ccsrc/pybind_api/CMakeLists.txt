file(GLOB_RECURSE _PYNATIVE_ALL_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        "*.cc"
        )

add_library(_mindspore_pynative_obj OBJECT ${_PYNATIVE_ALL_SRC_FILES})