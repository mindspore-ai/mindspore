file(GLOB_RECURSE _CURRENT_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")
set_property(SOURCE ${_CURRENT_SRC_FILES} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_MD)

if(${CMAKE_SYSTEM_NAME} MATCHES "Windows")
    include(${CMAKE_SOURCE_DIR}/mindspore/ccsrc/minddata/cmake/dataset.cmake)
    set(KERNELS_IR_VISION ${CMAKE_BINARY_DIR}/merge/mindspore/minddata/vision/transform)
    set(EXCLUDE_FILES "")
    if(NOT ENABLE_FFMPEG)
    endif()
    merge_minddata_files(${CMAKE_CURRENT_SOURCE_DIR}/ ${KERNELS_IR_VISION}
            dataset_vision_transform "${EXCLUDE_FILES}" "" TRUE)
    file(GLOB KERNELS_IR_VISION_LIST "${KERNELS_IR_VISION}/*.cc")
    add_library(vision-transform OBJECT ${KERNELS_IR_VISION_LIST})
else()
    set(DATASET_KERNELS_IR_VISION_SRC_FILES
            center_crop_ir.cc
            decode_ir.cc
            hwc_to_chw_ir.cc
            normalize_ir.cc
            random_color_adjust_ir.cc
            random_crop_decode_resize_ir.cc
            random_crop_ir.cc
            random_horizontal_flip_ir.cc
            random_resized_crop_ir.cc
            random_vertical_flip_ir.cc
            rescale_ir.cc
            resize_ir.cc
            resize_preserve_ar_ir.cc
            rgb_to_bgr_ir.cc
            rgb_to_gray_ir.cc
            rgba_to_bgr_ir.cc
            rgba_to_rgb_ir.cc
            rotate_ir.cc
            swap_red_blue_ir.cc
            to_tensor_ir.cc
            ascend_vision_ir.cc
            )

    add_library(vision-transform OBJECT ${DATASET_KERNELS_IR_VISION_SRC_FILES})
endif()
