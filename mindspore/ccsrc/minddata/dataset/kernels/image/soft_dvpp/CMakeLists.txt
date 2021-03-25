file(GLOB_RECURSE _CURRENT_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")
set_property(SOURCE ${_CURRENT_SRC_FILES} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_MD)
set_property(SOURCE ${_CURRENT_SRC_FILES} PROPERTY COMPILE_DEFINITIONS google=mindspore_private)
add_subdirectory(utils)
add_library(kernels-soft-dvpp-image OBJECT
            soft_dvpp_decode_resize_jpeg_op.cc
            soft_dvpp_decode_random_crop_resize_jpeg_op.cc)