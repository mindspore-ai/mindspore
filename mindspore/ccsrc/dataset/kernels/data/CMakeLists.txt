file(GLOB_RECURSE _CURRENT_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")
set_property(SOURCE ${_CURRENT_SRC_FILES} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_MD)
add_library(kernels-data OBJECT
        data_utils.cc
        one_hot_op.cc
        pad_end_op.cc
        type_cast_op.cc
        to_float16_op.cc
        fill_op.cc
        slice_op.cc
        mask_op.cc
        concatenate_op.cc
        )
