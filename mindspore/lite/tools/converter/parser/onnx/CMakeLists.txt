file(GLOB_RECURSE ONNX_SRC_LIST ${CMAKE_CURRENT_SOURCE_DIR}/*.cc)

add_library(onnx_parser_mid OBJECT
        ${ONNX_SRC_LIST}
        )
