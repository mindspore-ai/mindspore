file(GLOB_RECURSE _GRAPH_KERNEL_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        "*.cc"
        )

set_property(SOURCE ${_GRAPH_KERNEL_SRC_FILES}
  PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_GRAPH_KERNEL)
add_library(_mindspore_backend_common_graph_kernel_obj OBJECT ${_GRAPH_KERNEL_SRC_FILES})
