file(GLOB_RECURSE _CURRENT_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")
set_property(SOURCE ${_CURRENT_SRC_FILES} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_MD)

if(${CMAKE_SYSTEM_NAME} MATCHES "Windows")
    include(${CMAKE_SOURCE_DIR}/mindspore/ccsrc/minddata/cmake/dataset.cmake)
    set(KERNELS_IR_VISION ${CMAKE_BINARY_DIR}/merge/mindspore/minddata/vision/transform)
    set(EXCLUDE_FILES "")
    if(NOT ENABLE_FFMPEG)
        set(EXCLUDE_FILES "decode_video_ir.cc")
    endif()
    merge_minddata_files(${CMAKE_CURRENT_SOURCE_DIR}/ ${KERNELS_IR_VISION}
            dataset_vision_transform "${EXCLUDE_FILES}" "" TRUE)
    file(GLOB KERNELS_IR_VISION_LIST "${KERNELS_IR_VISION}/*.cc")
    add_library(vision-transform OBJECT ${KERNELS_IR_VISION_LIST})
else()
    set(DATASET_KERNELS_IR_VISION_SRC_FILES
            adjust_gamma_ir.cc
            auto_augment_ir.cc
            bounding_box_augment_ir.cc
            center_crop_ir.cc
            crop_ir.cc
            cutmix_batch_ir.cc
            decode_ir.cc
            erase_ir.cc
            hwc_to_chw_ir.cc
            mixup_batch_ir.cc
            normalize_ir.cc
            normalize_pad_ir.cc
            pad_ir.cc
            perspective_ir.cc
            posterize_ir.cc
            rand_augment_ir.cc
            random_color_adjust_ir.cc
            random_crop_decode_resize_ir.cc
            random_crop_ir.cc
            random_crop_with_bbox_ir.cc
            random_horizontal_flip_ir.cc
            random_horizontal_flip_with_bbox_ir.cc
            random_lighting_ir.cc
            random_posterize_ir.cc
            random_resized_crop_ir.cc
            random_resized_crop_with_bbox_ir.cc
            random_resize_with_bbox_ir.cc
            random_solarize_ir.cc
            random_vertical_flip_ir.cc
            random_vertical_flip_with_bbox_ir.cc
            rescale_ir.cc
            resize_ir.cc
            resize_preserve_ar_ir.cc
            resize_with_bbox_ir.cc
            rgb_to_bgr_ir.cc
            rgb_to_gray_ir.cc
            rgba_to_bgr_ir.cc
            rgba_to_rgb_ir.cc
            rotate_ir.cc
            solarize_ir.cc
            swap_red_blue_ir.cc
            to_tensor_ir.cc
            trivial_augment_wide_ir.cc
            ascend_vision_ir.cc
            )

    if(ENABLE_FFMPEG)
        add_library(vision-transform OBJECT ${DATASET_KERNELS_IR_VISION_SRC_FILES} decode_video_ir.cc)
    else()
        add_library(vision-transform OBJECT ${DATASET_KERNELS_IR_VISION_SRC_FILES})
    endif()
endif()
