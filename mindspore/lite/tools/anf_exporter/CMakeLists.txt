file(GLOB_RECURSE ANF_EXPORTER_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        *.cc
        )
add_library(anf_exporter_mid OBJECT
        ${ANF_EXPORTER_SRC_LIST}
        )
