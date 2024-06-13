
file(GLOB_RECURSE _AMP_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        "*.cc"
        )

set_property(SOURCE ${_AMP_SRC_LIST} PROPERTY COMPILE_DEFINITIONS
  SUBMODULE_ID=mindspore::SubModuleId::SM_COMMON)

add_library(_mindspore_common_amp_obj OBJECT ${_AMP_SRC_LIST})
