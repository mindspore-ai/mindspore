file(GLOB_RECURSE _PREDICT_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
    "predict.cc"
    "generator/utils/ir_model_util.cc"
    "converter/*.cc"
    "converter/attr_utils/*.cc"
    "converter/lite_model/*.cc"
    "converter/lite_model/operations/*.cc"
)

if (ENABLE_D)
    file(GLOB_RECURSE _D_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "generator/ir/*.cc")
    list(APPEND _PREDICT_SRC_LIST ${_D_SRC_LIST})
endif ()
add_library(_mindspore_predict_obj OBJECT ${_PREDICT_SRC_LIST})