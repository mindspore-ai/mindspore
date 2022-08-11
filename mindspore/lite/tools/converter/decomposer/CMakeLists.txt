file(GLOB DECOMPOSER
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cc
        )
set_property(SOURCE ${DECOMPOSER} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_LITE)
add_library(decomposer_mid OBJECT ${DECOMPOSER})

add_dependencies(decomposer_mid fbs_src)
add_dependencies(decomposer_mid fbs_inner_src)
