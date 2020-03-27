file(GLOB_RECURSE _ONNX_ALL_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        "*.cc"
        )

add_library(_mindspore_onnx_obj OBJECT ${_ONNX_ALL_SRC_FILES})
