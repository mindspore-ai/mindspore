if(ENABLE_AKG AND ${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    file(GLOB_RECURSE _GK_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
            "*.cc"
            )
    file(GLOB_RECURSE _GK_LITE_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
            "lite_adapter/*.cc"
            )
    list(REMOVE_ITEM _GK_SRC_LIST ${_GK_LITE_LIST})
    list(APPEND _GRAPH_KERNEL_SRC_FILES ${_GK_SRC_LIST})

    add_library(_mindspore_common_graph_kernel_obj OBJECT ${_GRAPH_KERNEL_SRC_FILES})
endif()
