file(GLOB_RECURSE CUSTOM_SRC RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")

add_library(_mindspore_ops_kernel_ascend_custom_obj OBJECT ${CUSTOM_SRC})
