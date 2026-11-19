file(GLOB_RECURSE _CURRENT_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")
set_property(SOURCE ${_CURRENT_SRC_FILES} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_MD)

if(NOT (CMAKE_SYSTEM_NAME MATCHES "Windows"))
        set(ABSL_DEPEND_FILES
                parse_example_op.cc)
endif()

add_library(general-kernels OBJECT
        concatenate_op.cc
        data_utils.cc
        slice_op.cc
        to_float16_op.cc
        type_cast_op.cc
        compose_op.cc
        ${ABSL_DEPEND_FILES}
        )
