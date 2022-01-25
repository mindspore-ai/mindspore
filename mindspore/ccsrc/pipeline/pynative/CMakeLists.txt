file(GLOB_RECURSE _PYNATIVE_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "pynative_execute.cc")

set_property(SOURCE ${_PYNATIVE_SRC_LIST} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_PYNATIVE)
add_library(_mindspore_pipeline_pynative_obj OBJECT ${_PYNATIVE_SRC_LIST})
