file(GLOB_RECURSE _CURRENT_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")
set_property(SOURCE ${_CURRENT_SRC_FILES} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_MD)
add_subdirectory(utils)
add_library(kernels-dvpp-image OBJECT
        dvpp_crop_jpeg_op.cc
        dvpp_decode_resize_crop_jpeg_op.cc
        dvpp_decode_resize_jpeg_op.cc
        dvpp_decode_jpeg_op.cc
        dvpp_decode_png_op.cc
        dvpp_normalize_op.cc
        dvpp_resize_jpeg_op.cc)
add_dependencies(kernels-dvpp-image dvpp-utils)