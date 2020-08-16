file(GLOB_RECURSE ANF_IMPORTER_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        *.cc
        )
add_library(anf_importer_mid OBJECT
        ${ANF_IMPORTER_SRC_LIST}
        )
