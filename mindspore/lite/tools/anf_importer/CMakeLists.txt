file(GLOB ANF_IMPORTER_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        *.cc
        )
set_property(SOURCE ${ANF_IMPORTER_SRC_LIST} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_LITE)
add_library(anf_importer_mid OBJECT
        ${ANF_IMPORTER_SRC_LIST}
        )
add_dependencies(anf_importer_mid proto_mid)

add_dependencies(anf_importer_mid fbs_src)
add_dependencies(anf_importer_mid fbs_inner_src)
