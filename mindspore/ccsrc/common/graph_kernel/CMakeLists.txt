file(GLOB_RECURSE _GK_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        "*.cc"
        )
list(APPEND _GRAPH_KERNEL_SRC_FILES ${_GK_SRC_LIST})

add_library(_mindspore_common_graph_kernel_obj OBJECT ${_GRAPH_KERNEL_SRC_FILES})
