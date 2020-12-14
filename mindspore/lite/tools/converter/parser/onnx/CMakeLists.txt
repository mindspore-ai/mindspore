file(GLOB_RECURSE ONNX_SRC_LIST ${CMAKE_CURRENT_SOURCE_DIR}/*.cc)
set_property(SOURCE ${ONNX_SRC_LIST} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_LITE)
add_library(onnx_parser_mid OBJECT
        ${ONNX_SRC_LIST}
        )
add_dependencies(onnx_parser_mid proto_mid)
add_dependencies(onnx_parser_mid fbs_src)
add_dependencies(onnx_parser_mid fbs_inner_src)
