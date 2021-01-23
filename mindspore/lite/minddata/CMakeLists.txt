find_package(Patch)
if(NOT Patch_FOUND)
  message(FATAL_ERROR "Patch not found, please set environment variable MS_PATCH_PATH to path where Patch is located, "
  "usually found in GIT_PATH/usr/bin on Windows")
endif()

include(${TOP_DIR}/cmake/external_libs/jpeg_turbo.cmake)

set(MINDDATA_DIR ${CCSRC_DIR}/minddata/dataset)
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -g2 -ggdb -fno-inline-functions -fno-omit-frame-pointer -D_LIBCPP_INLINE_VISIBILITY='' -D_LIBCPP_DISABLE_EXTERN_TEMPLATE=1 -DHALF_ENABLE_CPP11_USER_LITERALS=0 -D_FORTIFY_SOURCE=2 -Wno-cpp")
set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -Werror -Wno-return-std-move -Wno-unused-private-field -Wno-unused-lambda-capture -Wno-sign-compare -Wno-overloaded-virtual -Wno-unneeded-internal-declaration -Wno-unused-variable -Wno-pessimizing-move -Wno-inconsistent-missing-override")


set(CMAKE_CXX_FLAGS "$ENV{CXXFLAGS} -I/usr/local/include -std=c++17 -Wall -fPIC")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OPTION_CXX_FLAGS}")

if(PLATFORM_ARM)
    set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O2 -Werror -Wno-return-std-move -Wno-unused-private-field -Wno-unused-lambda-capture -Wno-sign-compare -Wno-overloaded-virtual -Wno-unneeded-internal-declaration -Wno-unused-variable -Wno-pessimizing-move -Wno-inconsistent-missing-override -DHALF_ENABLE_CPP11_USER_LITERALS=0 -D_FORTIFY_SOURCE=2")
else()
    set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O2 -Wl,--allow-shlib-undefined -DHALF_ENABLE_CPP11_USER_LITERALS=0 -D_FORTIFY_SOURCE=2")
endif()

set(CMAKE_C_FLAGS "-fPIC -fPIE -D_FORTIFY_SOURCE=2 -O2 -Wall -Werror -fstack-protector-strong -Wno-attributes -Wno-deprecated-declarations -Wno-missing-braces ${CMAKE_C_FLAGS}")
set(CMAKE_CXX_FLAGS "-fPIC -fPIE -D_FORTIFY_SOURCE=2 -O2 -Wall -Werror -fstack-protector-strong -Wno-attributes -Wno-deprecated-declarations -Wno-missing-braces -Wno-overloaded-virtual ${CMAKE_CXX_FLAGS}")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-non-virtual-dtor")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-sign-compare")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-private-field")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-reorder")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-switch")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-sequence-point")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-variable")
if(PLATFORM_ARM)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-delete-non-virtual-dtor")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-uninitialized")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-delete-non-abstract-non-virtual-dtor")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-maybe-uninitialized")
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-format")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-attributes")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=default")

set(CMAKE_SHARED_LINKER_FLAGS_RELEASE "${CMAKE_SHARED_LINKER_FLAGS} -s")

AUX_SOURCE_DIRECTORY(${MINDDATA_DIR}/api MINDDATA_API_SRC_FILES)

AUX_SOURCE_DIRECTORY(${MINDDATA_DIR}/callback MINDDATA_CALLBACK_SRC_FILES)

AUX_SOURCE_DIRECTORY(${MINDDATA_DIR}/core MINDDATA_CORE_SRC_FILES)

AUX_SOURCE_DIRECTORY(${MINDDATA_DIR}/engine MINDDATA_ENGINE_SRC_FILES)

# TODO: add cache
AUX_SOURCE_DIRECTORY(${MINDDATA_DIR}/engine/cache MINDDATA_ENGINE_CACHE_SRC_FILES)

AUX_SOURCE_DIRECTORY(${MINDDATA_DIR}/engine/consumers MINDDATA_ENGINE_CONSUMERS_SRC_FILES)
AUX_SOURCE_DIRECTORY(${MINDDATA_DIR}/engine/datasetops MINDDATA_ENGINE_DATASETOPS_SRC_FILES)
AUX_SOURCE_DIRECTORY(${MINDDATA_DIR}/engine/datasetops/map_op MINDDATA_ENGINE_DATASETOPS_MAPOP_SRC_FILES)
AUX_SOURCE_DIRECTORY(${MINDDATA_DIR}/engine/datasetops/source MINDDATA_ENGINE_DATASETOPS_SOURCE_SRC_FILES)
AUX_SOURCE_DIRECTORY(${MINDDATA_DIR}/engine/datasetops/source/sampler MINDDATA_ENGINE_DATASETOPS_SOURCE_SAMPLER_SRC_FILES)
AUX_SOURCE_DIRECTORY(${MINDDATA_DIR}/engine/ir/cache MINDDATA_ENGINE_IR_CACHE_SRC_FILES)
AUX_SOURCE_DIRECTORY(${MINDDATA_DIR}/engine/ir/datasetops MINDDATA_ENGINE_IR_DATASETOPS_SRC_FILES)
AUX_SOURCE_DIRECTORY(${MINDDATA_DIR}/engine/ir/datasetops/source MINDDATA_ENGINE_IR_DATASETOPS_SOURCE_SRC_FILES)

AUX_SOURCE_DIRECTORY(${MINDDATA_DIR}/engine/opt MINDDATA_ENGINE_OPT_SRC_FILES)
# AUX_SOURCE_DIRECTORY(${MINDDATA_DIR}/engine/opt/optional MINDDATA_ENGINE_OPT_OPTIONAL_SRC_FILES)
AUX_SOURCE_DIRECTORY(${MINDDATA_DIR}/engine/opt/post MINDDATA_ENGINE_OPT_POST_SRC_FILES)
AUX_SOURCE_DIRECTORY(${MINDDATA_DIR}/engine/opt/pre MINDDATA_ENGINE_OPT_PRE_SRC_FILES)
AUX_SOURCE_DIRECTORY(${MINDDATA_DIR}/engine/opt/util MINDDATA_ENGINE_OPT_UTIL_SRC_FILES)

AUX_SOURCE_DIRECTORY(${MINDDATA_DIR}/engine/perf MINDDATA_ENGINE_PERF_SRC_FILES)

AUX_SOURCE_DIRECTORY(${MINDDATA_DIR}/kernels MINDDATA_KERNELS_SRC_FILES)

AUX_SOURCE_DIRECTORY(${MINDDATA_DIR}/kernels/image MINDDATA_KERNELS_IMAGE_SRC_FILES)

AUX_SOURCE_DIRECTORY(${MINDDATA_DIR}/kernels/data MINDDATA_KERNELS_DATA_SRC_FILES)

# TODO: add text src
AUX_SOURCE_DIRECTORY(${MINDDATA_DIR}/text MINDDATA_TEXT_SRC_FILES)

AUX_SOURCE_DIRECTORY(${MINDDATA_DIR}/text/kernels MINDDATA_TEXT_KERNELS_SRC_FILES)

AUX_SOURCE_DIRECTORY(${MINDDATA_DIR}/util MINDDATA_UTIL_SRC_FILES)

AUX_SOURCE_DIRECTORY(${MINDDATA_DIR}/kernels/image/lite_cv  MINDDATA_KERNELS_IMAGE_LITE_CV_FILES)

if(BUILD_MINDDATA STREQUAL "full")
    set(BUILD_MINDDATA "wrapper")
endif()

if(BUILD_MINDDATA STREQUAL "full")
    include_directories("${CMAKE_SOURCE_DIR}/../ccsrc/minddata/dataset/kernels/image")
    list(REMOVE_ITEM MINDDATA_API_SRC_FILES
            "${MINDDATA_DIR}/api/text.cc"
            "${MINDDATA_DIR}/api/minddata_eager.cc"
            )

    list(REMOVE_ITEM MINDDATA_CALLBACK_SRC_FILES
            "${MINDDATA_DIR}/callback/py_ds_callback.cc"
            )

    list(REMOVE_ITEM MINDDATA_CORE_SRC_FILES
            "${MINDDATA_DIR}/core/cv_tensor.cc"
            )

    list(REMOVE_ITEM MINDDATA_KERNELS_SRC_FILES "${MINDDATA_DIR}/kernels/py_func_op.cc")
    list(REMOVE_ITEM MINDDATA_ENGINE_DATASETOPS_SRC_FILES
            "${MINDDATA_DIR}/engine/datasetops/build_sentence_piece_vocab_op.cc"
            "${MINDDATA_DIR}/engine/datasetops/filter_op.cc"
            "${MINDDATA_DIR}/engine/datasetops/barrier_op.cc"
            "${MINDDATA_DIR}/engine/datasetops/bucket_batch_by_length_op.cc"
            "${MINDDATA_DIR}/engine/datasetops/build_vocab_op.cc"
            "${MINDDATA_DIR}/engine/datasetops/cache_merge_op.cc"
            "${MINDDATA_DIR}/engine/datasetops/cache_base_op.cc"
            "${MINDDATA_DIR}/engine/datasetops/cache_lookup_op.cc"
            "${MINDDATA_DIR}/engine/datasetops/cache_op.cc"
            "${MINDDATA_DIR}/engine/datasetops/concat_op.cc"
            "${MINDDATA_DIR}/engine/datasetops/rename_op.cc"
            "${MINDDATA_DIR}/engine/datasetops/skip_op.cc"
            "${MINDDATA_DIR}/engine/datasetops/take_op.cc"
            "${MINDDATA_DIR}/engine/datasetops/zip_op.cc"
            )

    list(REMOVE_ITEM MINDDATA_ENGINE_DATASETOPS_SOURCE_SRC_FILES
            "${MINDDATA_DIR}/engine/datasetops/source/generator_op.cc"
            "${MINDDATA_DIR}/engine/datasetops/source/voc_op.cc"
            "${MINDDATA_DIR}/engine/datasetops/source/manifest_op.cc"
            "${MINDDATA_DIR}/engine/datasetops/source/mindrecord_op.cc"
            "${MINDDATA_DIR}/engine/datasetops/source/tf_reader_op.cc"
            "${MINDDATA_DIR}/engine/datasetops/source/celeba_op.cc"
            "${MINDDATA_DIR}/engine/datasetops/source/cifar_op.cc"
            "${MINDDATA_DIR}/engine/datasetops/source/clue_op.cc"
            "${MINDDATA_DIR}/engine/datasetops/source/coco_op.cc"
            "${MINDDATA_DIR}/engine/datasetops/source/csv_op.cc"
            "${MINDDATA_DIR}/engine/datasetops/source/image_folder_op.cc"
            "${MINDDATA_DIR}/engine/datasetops/source/mnist_op.cc"
            "${MINDDATA_DIR}/engine/datasetops/source/random_data_op.cc"
            "${MINDDATA_DIR}/engine/datasetops/source/text_file_op.cc"
            "${MINDDATA_DIR}/engine/datasetops/source/voc_op.cc"
            )

    list(REMOVE_ITEM MINDDATA_ENGINE_DATASETOPS_SOURCE_SAMPLER_SRC_FILES
            "${MINDDATA_DIR}/engine/datasetops/source/sampler/python_sampler.cc"
            )

    list(REMOVE_ITEM MINDDATA_ENGINE_OPT_POST_SRC_FILES
            "${MINDDATA_DIR}/engine/opt/post/generator_node_pass.cc"
            )

    list(REMOVE_ITEM MINDDATA_ENGINE_OPT_POST_SRC_FILES
            "${MINDDATA_DIR}/engine/opt/post/repeat_pass.cc"
            )

    list(REMOVE_ITEM MINDDATA_ENGINE_OPT_PRE_SRC_FILES
            "${MINDDATA_DIR}/engine/opt/pre/cache_transform_pass.cc"
            "${MINDDATA_DIR}/engine/opt/pre/cache_error_pass.cc"
            )

    list(REMOVE_ITEM MINDDATA_ENGINE_IR_CACHE_SRC_FILES
            "${MINDDATA_DIR}/engine/ir/cache/dataset_cache_impl.cc"
            "${MINDDATA_DIR}/engine/ir/cache/pre_built_dataset_cache.cc"
            )

    list(REMOVE_ITEM MINDDATA_ENGINE_IR_DATASETOPS_SOURCE_SRC_FILES
            "${MINDDATA_DIR}/engine/ir/datasetops/source/generator_node.cc"
            "${MINDDATA_DIR}/engine/ir/datasetops/source/manifest_node.cc"
            "${MINDDATA_DIR}/engine/ir/datasetops/source/minddata_node.cc"
            "${MINDDATA_DIR}/engine/ir/datasetops/source/tf_record_node.cc"
            "${MINDDATA_DIR}/engine/ir/datasetops/source/voc_node.cc"
            "${MINDDATA_DIR}/engine/ir/datasetops/source/celeba_node.cc"
            "${MINDDATA_DIR}/engine/ir/datasetops/source/cifar10_node.cc"
            "${MINDDATA_DIR}/engine/ir/datasetops/source/cifar100_node.cc"
            "${MINDDATA_DIR}/engine/ir/datasetops/source/coco_node.cc"
            "${MINDDATA_DIR}/engine/ir/datasetops/source/csv_node.cc"
            "${MINDDATA_DIR}/engine/ir/datasetops/source/image_folder_node.cc"
            "${MINDDATA_DIR}/engine/ir/datasetops/source/manifest_node.cc"
            "${MINDDATA_DIR}/engine/ir/datasetops/source/mnist_node.cc"
            "${MINDDATA_DIR}/engine/ir/datasetops/source/random_node.cc"
            "${MINDDATA_DIR}/engine/ir/datasetops/source/text_file_node.cc"
            "${MINDDATA_DIR}/engine/ir/datasetops/source/clue_node.cc"
            )
    list(REMOVE_ITEM MINDDATA_KERNELS_IMAGE_SRC_FILES
            "${MINDDATA_DIR}/kernels/image/affine_op.cc"
            "${MINDDATA_DIR}/kernels/image/auto_contrast_op.cc"
            "${MINDDATA_DIR}/kernels/image/bounding_box_op.cc"
            "${MINDDATA_DIR}/kernels/image/bounding_box_augment_op.cc"
            "${MINDDATA_DIR}/kernels/image/concatenate_op.cc"
            "${MINDDATA_DIR}/kernels/image/cut_out_op.cc"
            "${MINDDATA_DIR}/kernels/image/cutmix_batch_op.cc"
            "${MINDDATA_DIR}/kernels/image/equalize_op.cc"
            "${MINDDATA_DIR}/kernels/image/hwc_to_chw_op.cc"
            "${MINDDATA_DIR}/kernels/image/image_utils.cc"
            "${MINDDATA_DIR}/kernels/image/invert_op.cc"
            "${MINDDATA_DIR}/kernels/image/math_utils.cc"
            "${MINDDATA_DIR}/kernels/image/mixup_batch_op.cc"
            "${MINDDATA_DIR}/kernels/image/pad_op.cc"
            "${MINDDATA_DIR}/kernels/image/posterize_op.cc"
            "${MINDDATA_DIR}/kernels/image/normalize_pad_op.cc"
            "${MINDDATA_DIR}/kernels/image/random_affine_op.cc"
            "${MINDDATA_DIR}/kernels/image/random_color_adjust_op.cc"
            "${MINDDATA_DIR}/kernels/image/random_crop_and_resize_with_bbox_op.cc"
            "${MINDDATA_DIR}/kernels/image/random_crop_decode_resize_op.cc"
            "${MINDDATA_DIR}/kernels/image/random_crop_and_resize_op.cc"
            "${MINDDATA_DIR}/kernels/image/random_crop_op.cc"
            "${MINDDATA_DIR}/kernels/image/random_crop_with_bbox_op.cc"
            "${MINDDATA_DIR}/kernels/image/random_horizontal_flip_op.cc"
            "${MINDDATA_DIR}/kernels/image/random_horizontal_flip_with_bbox_op.cc"
            "${MINDDATA_DIR}/kernels/image/random_posterize_op.cc"
            "${MINDDATA_DIR}/kernels/image/random_resize_op.cc"
            "${MINDDATA_DIR}/kernels/image/random_rotation_op.cc"
            "${MINDDATA_DIR}/kernels/image/random_select_subpolicy_op.cc"
            "${MINDDATA_DIR}/kernels/image/random_solarize_op.cc"
            "${MINDDATA_DIR}/kernels/image/random_vertical_flip_op.cc"
            "${MINDDATA_DIR}/kernels/image/random_vertical_flip_with_bbox_op.cc"
            "${MINDDATA_DIR}/kernels/image/random_sharpness_op.cc"
            "${MINDDATA_DIR}/kernels/image/rescale_op.cc"
            "${MINDDATA_DIR}/kernels/image/rgba_to_bgr_op.cc"
            "${MINDDATA_DIR}/kernels/image/rgba_to_rgb_op.cc"
            "${MINDDATA_DIR}/kernels/image/sharpness_op.cc"
            "${MINDDATA_DIR}/kernels/image/solarize_op.cc"
            "${MINDDATA_DIR}/kernels/image/swap_red_blue_op.cc"
            "${MINDDATA_DIR}/kernels/image/uniform_aug_op.cc"
            "${MINDDATA_DIR}/kernels/image/resize_with_bbox_op.cc"
            "${MINDDATA_DIR}/kernels/image/random_resize_with_bbox_op.cc"
            "${MINDDATA_DIR}/kernels/image/random_color_op.cc"
            )

    list(REMOVE_ITEM MINDDATA_ENGINE_IR_DATASETOPS_SRC_FILES
            "${MINDDATA_DIR}/engine/ir/datasetops/bucket_batch_by_length_node.cc"
            "${MINDDATA_DIR}/engine/ir/datasetops/build_sentence_piece_vocab_node.cc"
            "${MINDDATA_DIR}/engine/ir/datasetops/build_vocab_node.cc"
            "${MINDDATA_DIR}/engine/ir/datasetops/filter_node.cc"
            "${MINDDATA_DIR}/engine/ir/datasetops/sync_wait_node.cc"
            "${MINDDATA_DIR}/engine/ir/datasetops/skip_node.cc"
            "${MINDDATA_DIR}/engine/ir/datasetops/take_node.cc"
            "${MINDDATA_DIR}/engine/ir/datasetops/transfer_node.cc"
            "${MINDDATA_DIR}/engine/ir/datasetops/zip_node.cc"
            "${MINDDATA_DIR}/engine/ir/datasetops/rename_node.cc"
            "${MINDDATA_DIR}/engine/ir/datasetops/concat_node.cc"
            )
    list(REMOVE_ITEM MINDDATA_ENGINE_CONSUMERS_SRC_FILES
            "${MINDDATA_DIR}/engine/consumers/python_tree_consumer.cc"
            )

    list(REMOVE_ITEM MINDDATA_ENGINE_SRC_FILES
            "${MINDDATA_DIR}/engine/python_runtime_context.cc"
            )

    list(REMOVE_ITEM MINDDATA_KERNELS_DATA_SRC_FILES
            "${MINDDATA_DIR}/kernels/data/unique_op.cc"
            )

    list(REMOVE_ITEM MINDDATA_UTIL_SRC_FILES
            "${MINDDATA_DIR}/util/numa_interface.cc"
            )
    include_directories("${CMAKE_BINARY_DIR}/minddata/dataset/engine/cache")

    if(BUILD_MINDDATA_EXAMPLE AND (PLATFORM_ARM32 OR PLATFORM_ARM64))
        set(MINDDATA_TODAPI_SRC ${CMAKE_CURRENT_SOURCE_DIR}/wrapper/MDToDApi.cc)
    endif()

    add_library(minddata-lite SHARED
            ${MINDDATA_API_SRC_FILES}
            ${MINDDATA_CALLBACK_SRC_FILES}
            ${MINDDATA_CORE_SRC_FILES}
            ${MINDDATA_ENGINE_SRC_FILES}
            #${MINDDATA_ENGINE_CACHE_SRC_FILES}
            ${MINDDATA_ENGINE_CONSUMERS_SRC_FILES}
            ${MINDDATA_ENGINE_DATASETOPS_SRC_FILES}
            ${MINDDATA_ENGINE_DATASETOPS_MAPOP_SRC_FILES}
            ${MINDDATA_ENGINE_DATASETOPS_SOURCE_SRC_FILES}
            ${MINDDATA_ENGINE_DATASETOPS_SOURCE_SAMPLER_SRC_FILES}
            ${MINDDATA_ENGINE_IR_DATASETOPS_SRC_FILES}
            ${MINDDATA_ENGINE_IR_CACHE_SRC_FILES}
            ${MINDDATA_ENGINE_IR_DATASETOPS_SOURCE_SRC_FILES}
            ${MINDDATA_ENGINE_OPT_SRC_FILES}
            ${MINDDATA_ENGINE_OPT_OPTIONAL_SRC_FILES}
            ${MINDDATA_ENGINE_OPT_POST_SRC_FILES}
            ${MINDDATA_ENGINE_OPT_PRE_SRC_FILES}
            ${MINDDATA_ENGINE_OPT_UTIL_SRC_FILES}
            ${MINDDATA_ENGINE_PERF_SRC_FILES}
            ${MINDDATA_KERNELS_SRC_FILES}
            ${MINDDATA_KERNELS_IMAGE_LITE_CV_FILES}
            ${MINDDATA_KERNELS_IMAGE_SRC_FILES}
            ${MINDDATA_KERNELS_DATA_SRC_FILES}
            ${MINDDATA_UTIL_SRC_FILES}
            ${MINDDATA_EXAMPLE_SRC}
            ${CMAKE_CURRENT_SOURCE_DIR}/../src/common/log_adapter.cc
            ${CORE_DIR}/utils/ms_utils.cc
            )

    find_package(Threads REQUIRED)
    target_link_libraries(minddata-lite
            securec
            mindspore::jpeg_turbo
            mindspore::turbojpeg
            mindspore::json
            Threads::Threads
            )

    # ref: https://github.com/android/ndk/issues/1202
    if(PLATFORM_ARM32)
        file(GLOB_RECURSE LIBCLANG_RT_LIB $ENV{ANDROID_NDK}/libclang_rt.builtins-arm-android.a)
        if(LIBCLANG_RT_LIB STREQUAL "")
            MESSAGE(FATAL_ERROR "Cannot find libclang_rt.builtins-arm-androi2d.a in $ENV{ANDROID_NDK}")
        endif()
        target_link_libraries(minddata-lite ${LIBCLANG_RT_LIB})
    endif()

    if(PLATFORM_ARM32 OR PLATFORM_ARM64)
        target_link_libraries(minddata-lite log)
    elseif(BUILD_MINDDATA_EXAMPLE)
    endif()
elseif(BUILD_MINDDATA STREQUAL "wrapper")
    include_directories("${MINDDATA_DIR}/kernels/image")
    include_directories("${MINDDATA_DIR}/util")
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/wrapper)
    set(MINDDATA_TODAPI_SRC
            ${MINDDATA_DIR}/core/tensor_shape.cc
            ${MINDDATA_DIR}/core/tensor.cc
            ${MINDDATA_DIR}/core/config_manager.cc
            ${MINDDATA_DIR}/core/data_type.cc
            ${MINDDATA_DIR}/core/tensor_helpers.cc
            ${MINDDATA_DIR}/core/global_context.cc
            ${MINDDATA_DIR}/core/tensor_row.cc
            ${MINDDATA_DIR}/api/vision.cc
            ${MINDDATA_DIR}/api/execute.cc
            ${MINDDATA_DIR}/api/transforms.cc
            ${MINDDATA_DIR}/api/de_tensor.cc
            ${MINDDATA_DIR}/util/path.cc
            ${MINDDATA_DIR}/util/status.cc
            ${MINDDATA_DIR}/util/data_helper.cc
            ${MINDDATA_DIR}/util/memory_pool.cc
            ${MINDDATA_DIR}/engine/data_schema.cc
            ${MINDDATA_DIR}/kernels/tensor_op.cc
            ${MINDDATA_DIR}/kernels/image/lite_image_utils.cc
            ${MINDDATA_DIR}/kernels/image/center_crop_op.cc
            ${MINDDATA_DIR}/kernels/image/crop_op.cc
            ${MINDDATA_DIR}/kernels/image/normalize_op.cc
            ${MINDDATA_DIR}/kernels/image/resize_op.cc
            ${MINDDATA_DIR}/kernels/image/rotate_op.cc
            ${MINDDATA_DIR}/kernels/data/compose_op.cc
            ${MINDDATA_DIR}/kernels/data/duplicate_op.cc
            ${MINDDATA_DIR}/kernels/data/one_hot_op.cc
            ${MINDDATA_DIR}/kernels/data/random_apply_op.cc
            ${MINDDATA_DIR}/kernels/data/random_choice_op.cc
            ${MINDDATA_DIR}/kernels/data/type_cast_op.cc
            ${MINDDATA_DIR}/kernels/data/data_utils.cc
            ${MINDDATA_DIR}/kernels/image/exif_utils.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/wrapper/MDToDApi.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/wrapper/album_op_android.cc
            )

    add_library(minddata-lite SHARED
            ${MINDDATA_KERNELS_IMAGE_LITE_CV_FILES}
            ${CMAKE_CURRENT_SOURCE_DIR}/../src/common/log_adapter.cc
            ${CORE_DIR}/utils/ms_utils.cc
            ${MINDDATA_TODAPI_SRC}
            )

    find_package(Threads REQUIRED)
    target_link_libraries(minddata-lite
            securec
            mindspore::jpeg_turbo
            mindspore::turbojpeg
            mindspore::json
            Threads::Threads
            )

    # ref: https://github.com/android/ndk/issues/1202
    if(PLATFORM_ARM32)
        file(GLOB_RECURSE LIBCLANG_RT_LIB $ENV{ANDROID_NDK}/libclang_rt.builtins-arm-android.a)
        if(LIBCLANG_RT_LIB STREQUAL "")
            MESSAGE(FATAL_ERROR "Cannot find libclang_rt.builtins-arm-androi2d.a in $ENV{ANDROID_NDK}")
        endif()
        target_link_libraries(minddata-lite ${LIBCLANG_RT_LIB})
    endif()

    if(PLATFORM_ARM32 OR PLATFORM_ARM64)
        target_link_libraries(minddata-lite log)
    elseif(BUILD_MINDDATA_EXAMPLE)
    endif()
elseif(BUILD_MINDDATA STREQUAL "lite")
    list(REMOVE_ITEM MINDDATA_CORE_SRC_FILES "${MINDDATA_DIR}/core/client.cc")
    list(REMOVE_ITEM MINDDATA_KERNELS_SRC_FILES "${MINDDATA_DIR}/kernels/py_func_op.cc")
    add_library(minddata_eager_mid OBJECT
        ${MINDDATA_DIR}/api/de_tensor.cc
        ${MINDDATA_DIR}/api/execute.cc
        )
    list(REMOVE_ITEM MINDDATA_CORE_SRC_FILES
        "${MINDDATA_DIR}/core/cv_tensor.cc"
        )

    list(REMOVE_ITEM MINDDATA_KERNELS_IMAGE_SRC_FILES
        "${MINDDATA_DIR}/kernels/image/affine_op.cc"
        "${MINDDATA_DIR}/kernels/image/auto_contrast_op.cc"
        "${MINDDATA_DIR}/kernels/image/bounding_box_op.cc"
        "${MINDDATA_DIR}/kernels/image/bounding_box_augment_op.cc"
        "${MINDDATA_DIR}/kernels/image/center_crop_op.cc"
        "${MINDDATA_DIR}/kernels/image/concatenate_op.cc"
        "${MINDDATA_DIR}/kernels/image/cut_out_op.cc"
        "${MINDDATA_DIR}/kernels/image/cutmix_batch_op.cc"
        "${MINDDATA_DIR}/kernels/image/equalize_op.cc"
        "${MINDDATA_DIR}/kernels/image/hwc_to_chw_op.cc"
        "${MINDDATA_DIR}/kernels/image/image_utils.cc"
        "${MINDDATA_DIR}/kernels/image/invert_op.cc"
        "${MINDDATA_DIR}/kernels/image/math_utils.cc"
        "${MINDDATA_DIR}/kernels/image/mixup_batch_op.cc"
        "${MINDDATA_DIR}/kernels/image/pad_op.cc"
        "${MINDDATA_DIR}/kernels/image/posterize_op.cc"
        "${MINDDATA_DIR}/kernels/image/random_affine_op.cc"
        "${MINDDATA_DIR}/kernels/image/random_color_adjust_op.cc"
        "${MINDDATA_DIR}/kernels/image/random_crop_and_resize_with_bbox_op.cc"
        "${MINDDATA_DIR}/kernels/image/random_crop_decode_resize_op.cc"
        "${MINDDATA_DIR}/kernels/image/random_crop_and_resize_op.cc"
        "${MINDDATA_DIR}/kernels/image/random_crop_op.cc"
        "${MINDDATA_DIR}/kernels/image/random_crop_with_bbox_op.cc"
        "${MINDDATA_DIR}/kernels/image/random_horizontal_flip_op.cc"
        "${MINDDATA_DIR}/kernels/image/random_horizontal_flip_with_bbox_op.cc"
        "${MINDDATA_DIR}/kernels/image/random_posterize_op.cc"
        "${MINDDATA_DIR}/kernels/image/random_resize_op.cc"
        "${MINDDATA_DIR}/kernels/image/random_rotation_op.cc"
        "${MINDDATA_DIR}/kernels/image/random_select_subpolicy_op.cc"
        "${MINDDATA_DIR}/kernels/image/random_solarize_op.cc"
        "${MINDDATA_DIR}/kernels/image/random_vertical_flip_op.cc"
        "${MINDDATA_DIR}/kernels/image/random_vertical_flip_with_bbox_op.cc"
        "${MINDDATA_DIR}/kernels/image/random_sharpness_op.cc"
        "${MINDDATA_DIR}/kernels/image/rescale_op.cc"
        "${MINDDATA_DIR}/kernels/image/rgba_to_bgr_op.cc"
        "${MINDDATA_DIR}/kernels/image/rgba_to_rgb_op.cc"
        "${MINDDATA_DIR}/kernels/image/sharpness_op.cc"
        "${MINDDATA_DIR}/kernels/image/solarize_op.cc"
        "${MINDDATA_DIR}/kernels/image/swap_red_blue_op.cc"
        "${MINDDATA_DIR}/kernels/image/uniform_aug_op.cc"
        "${MINDDATA_DIR}/kernels/image/resize_with_bbox_op.cc"
        "${MINDDATA_DIR}/kernels/image/random_resize_with_bbox_op.cc"
        "${MINDDATA_DIR}/kernels/image/random_color_op.cc"
        )

    add_library(minddata-lite SHARED
        ${MINDDATA_CORE_SRC_FILES}
        ${MINDDATA_KERNELS_SRC_FILES}
        ${MINDDATA_KERNELS_IMAGE_LITE_CV_FILES}
        ${MINDDATA_KERNELS_IMAGE_SRC_FILES}
        ${MINDDATA_KERNELS_DATA_SRC_FILES}
        ${MINDDATA_DIR}/util/status.cc
        ${MINDDATA_DIR}/util/memory_pool.cc
        ${MINDDATA_DIR}/util/path.cc
        ${MINDDATA_DIR}/api/transforms.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../src/common/log_adapter.cc
        ${CORE_DIR}/utils/ms_utils.cc
        )

    target_link_libraries(minddata-lite
        securec
        jpeg-turbo
        jpeg
        mindspore::json
        )

    # ref: https://github.com/android/ndk/issues/1202
    if(PLATFORM_ARM32)
        file(GLOB_RECURSE LIBCLANG_RT_LIB $ENV{ANDROID_NDK}/libclang_rt.builtins-arm-android.a)
        if(LIBCLANG_RT_LIB STREQUAL "")
            MESSAGE(FATAL_ERROR "Cannot find libclang_rt.builtins-arm-androi2d.a in $ENV{ANDROID_NDK}")
        endif()
        target_link_libraries(minddata-lite ${LIBCLANG_RT_LIB})
    endif()
elseif(BUILD_MINDDATA STREQUAL "lite_cv")
    include_directories(${MINDDATA_DIR}/kernels/image)
    message(STATUS ${MINDDATA_DIR}/kernels/image)
    add_library(minddata-lite SHARED
        ${MINDDATA_KERNELS_IMAGE_LITE_CV_FILES}
        )

    # ref: https://github.com/android/ndk/issues/1202
    if(PLATFORM_ARM32)
        file(GLOB_RECURSE LIBCLANG_RT_LIB $ENV{ANDROID_NDK}/libclang_rt.builtins-arm-android.a)
        if(LIBCLANG_RT_LIB STREQUAL "")
            MESSAGE(FATAL_ERROR "Cannot find libclang_rt.builtins-arm-androi2d.a in $ENV{ANDROID_NDK}")
        endif()
        target_link_libraries(minddata-lite ${LIBCLANG_RT_LIB})
    endif()
endif()
