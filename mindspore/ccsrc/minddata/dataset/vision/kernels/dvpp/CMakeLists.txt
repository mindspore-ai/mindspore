file(GLOB_RECURSE _CURRENT_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")
set_property(SOURCE ${_CURRENT_SRC_FILES} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_MD)
add_definitions(-DENABLE_DVPP_INTERFACE)
set(DVPP_IMAGE_SOURCE
        # Ascend310
        ascend310/dvpp_crop_jpeg_op.cc
        ascend310/dvpp_decode_resize_crop_jpeg_op.cc
        ascend310/dvpp_decode_resize_jpeg_op.cc
        ascend310/dvpp_decode_jpeg_op.cc
        ascend310/dvpp_decode_png_op.cc
        ascend310/dvpp_decode_video_op.cc
        ascend310/dvpp_normalize_op.cc
        ascend310/dvpp_resize_jpeg_op.cc
        # adaptor
        acl_adapter.cc
        )

if(ENABLE_D)
    include(${CMAKE_SOURCE_DIR}/cmake/metadef_variables.cmake)
set(DVPP_IMAGE_SOURCE
        ${DVPP_IMAGE_SOURCE}
        # Ascend910B
        ascend910b/dvpp_decode_op.cc
        ascend910b/dvpp_normalize_v2_op.cc
        ascend910b/dvpp_resize_op.cc
        ascend910b/dvpp_rotate_op.cc
        )
endif()

add_library(vision-kernels-dvpp-image OBJECT ${DVPP_IMAGE_SOURCE})
if(ENABLE_ACL)
    add_subdirectory(utils)
endif()
