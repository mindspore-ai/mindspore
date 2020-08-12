file(GLOB_RECURSE ANF_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        *.cc
        )
list(REMOVE_ITEM ANF_SRC_LIST import_from_meta_graph.cc)
add_library(anf_importer_mid OBJECT
        ${ANF_SRC_LIST}
        )
