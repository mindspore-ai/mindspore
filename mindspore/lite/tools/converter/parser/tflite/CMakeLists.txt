file(GLOB_RECURSE TFLITE_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        *.cc
        )
add_library(tflite_parser_mid OBJECT
        ${TFLITE_SRC_LIST}
        )
