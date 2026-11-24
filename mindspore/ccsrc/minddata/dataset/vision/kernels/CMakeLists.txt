file(GLOB_RECURSE _CURRENT_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")
set_property(SOURCE ${_CURRENT_SRC_FILES} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_MD)

add_subdirectory(lite_cv)
add_subdirectory(dvpp)

if(${CMAKE_SYSTEM_NAME} MATCHES "Windows")
    include(${CMAKE_SOURCE_DIR}/mindspore/ccsrc/minddata/cmake/dataset.cmake)
    set(KERNELS_IMAGE ${CMAKE_BINARY_DIR}/merge/mindspore/minddata/vision/kernels)
    set(EXCLUDE_FILES "")
    if(EXCLUDE_FILES)
        set(EXCLUDE_FILES "${EXCLUDE_FILES}|lite_image_utils.cc")
    else()
        set(EXCLUDE_FILES "lite_image_utils.cc")
    endif()
    merge_minddata_files(${CMAKE_CURRENT_SOURCE_DIR}/ ${KERNELS_IMAGE} dataset_vision_kernels
        "${EXCLUDE_FILES}" "pyav" TRUE)
    file(GLOB KERNELS_IMAGE_LIST "${KERNELS_IMAGE}/*.cc")
    add_library(vision-kernels OBJECT ${KERNELS_IMAGE_LIST})
else()
    if(ENABLE_FFMPEG)
        set(FFMPEG_DEPEND_FILES
            pyav/container.cc
            pyav/context.cc
            pyav/format.cc
            pyav/frame.cc
            pyav/packet.cc
            pyav/stream.cc
            video_utils.cc)
    endif()

    if(ENABLE_D)
        include(${CMAKE_SOURCE_DIR}/cmake/metadef_variables.cmake)
    endif()

    add_library(vision-kernels OBJECT
        affine_op.cc
        bounding_box.cc
        center_crop_op.cc
        decode_op.cc
        hwc_to_chw_op.cc
        image_utils.cc
        invert_op.cc
        math_utils.cc
        normalize_op.cc
        posterize_op.cc
        random_color_adjust_op.cc
        random_crop_decode_resize_op.cc
        random_crop_and_resize_with_bbox_op.cc
        random_crop_and_resize_op.cc
        random_crop_op.cc
        random_horizontal_flip_op.cc
        random_vertical_flip_op.cc
        rescale_op.cc
        resize_op.cc
        resize_preserve_ar_op.cc
        rgb_to_bgr_op.cc
        rgb_to_gray_op.cc
        rgba_to_bgr_op.cc
        rgba_to_rgb_op.cc
        sharpness_op.cc
        swap_red_blue_op.cc
        rotate_op.cc
        resize_cubic_op.cc
        to_tensor_op.cc
        ${FFMPEG_DEPEND_FILES})
endif()
add_dependencies(vision-kernels vision-kernels-dvpp-image)
