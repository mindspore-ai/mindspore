file(GLOB_RECURSE _CURRENT_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")
set_property(SOURCE ${_CURRENT_SRC_FILES} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_MD)
if(NOT (CMAKE_SYSTEM_NAME MATCHES "Windows"))
        set(ABSL_DEPEND_FILES
                parse_example_op.cc)
endif()
add_library(kernels-data OBJECT
        concatenate_op.cc
        data_utils.cc
        duplicate_op.cc
        fill_op.cc
        mask_op.cc
        one_hot_op.cc
        pad_end_op.cc
        slice_op.cc
        to_float16_op.cc
        type_cast_op.cc
        unique_op.cc
        ${ABSL_DEPEND_FILES}
        )
