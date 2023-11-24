file(GLOB_RECURSE _PYNATIVE_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        "forward/*.cc"
        "grad/*.cc"
        "op_function/*.cc"
        "op_function/auto_generate/*.cc"
        "*.cc"
)

set_property(SOURCE ${_PYNATIVE_SRC_LIST} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_PYNATIVE)
add_library(_mindspore_pipeline_pynative_obj OBJECT ${_PYNATIVE_SRC_LIST})
target_link_libraries(_mindspore_pipeline_pynative_obj mindspore_core _mindspore_kernel_obj)
add_dependencies(_mindspore_pipeline_pynative_obj generated_code)
