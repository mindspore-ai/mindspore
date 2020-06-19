file(GLOB_RECURSE _CURRENT_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")
set_property(SOURCE ${_CURRENT_SRC_FILES} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_MD)
add_library(kernels-image OBJECT
    center_crop_op.cc
    cut_out_op.cc
    decode_op.cc
    hwc_to_chw_op.cc
    image_utils.cc
    normalize_op.cc
    pad_op.cc
    random_color_adjust_op.cc
    random_crop_decode_resize_op.cc
    random_crop_and_resize_with_bbox_op.cc
    random_crop_and_resize_op.cc
    random_crop_op.cc
    random_crop_with_bbox_op.cc
    random_horizontal_flip_op.cc
        random_horizontal_flip_bbox_op.cc
        bounding_box_augment_op.cc
    random_resize_op.cc
    random_rotation_op.cc
    random_vertical_flip_op.cc
    random_vertical_flip_with_bbox_op.cc
    rescale_op.cc
    resize_bilinear_op.cc
    resize_op.cc
    uniform_aug_op.cc
    )
