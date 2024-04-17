file(GLOB_RECURSE _PIJIT_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        "graph_capture/*.cc"
        "graph_compiler/*.cc"
        "graph_guard/*.cc"
        "graph_build/*.cc"
        "utils/*.cc"
        "*.cc"
)

set_property(SOURCE ${_PIJIT_SRC_LIST} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_PI)
add_library(_mindspore_pipeline_jit_pi_obj OBJECT ${_PIJIT_SRC_LIST})
