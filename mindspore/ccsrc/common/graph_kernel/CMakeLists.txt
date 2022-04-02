if(ENABLE_AKG AND ${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    file(GLOB_RECURSE _GK_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
            "*.cc"
            )
    list(APPEND _GRAPH_KERNEL_SRC_FILES ${_GK_SRC_LIST})

    add_library(_mindspore_common_graph_kernel_obj OBJECT ${_GRAPH_KERNEL_SRC_FILES})
else()
    # GraphKernel only supports linux system now, but the GraphKernelFlags is a public class
    set(_GRAPH_KERNEL_PUBLIC_FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/graph_kernel_flags.cc
        )
    add_library(_mindspore_common_graph_kernel_obj OBJECT ${_GRAPH_KERNEL_PUBLIC_FILES})
endif()
