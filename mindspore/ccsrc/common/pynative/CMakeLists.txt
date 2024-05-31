file(GLOB_RECURSE _PYNATIVE_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")
add_library(_mindspore_common_pynative_obj OBJECT ${_PYNATIVE_SRC_LIST})
