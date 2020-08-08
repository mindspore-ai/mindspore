file(GLOB_RECURSE ANF_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        *.cc
        )
add_library(anf_exporter_mid OBJECT
        ${ANF_SRC_LIST}
        )

