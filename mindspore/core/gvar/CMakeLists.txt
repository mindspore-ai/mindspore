file(GLOB_RECURSE MS_GVAR_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cc)
set_property(SOURCE ${MS_GVAR_SRC_LIST} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_COMMON)
add_library(mindspore_gvar OBJECT ${MS_GVAR_SRC_LIST})
target_compile_definitions(mindspore_gvar PRIVATE BUILDING_DLL)
