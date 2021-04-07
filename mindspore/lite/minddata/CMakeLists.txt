find_package(Patch)
if(NOT Patch_FOUND)
  message(FATAL_ERROR "Patch not found, please set environment variable MS_PATCH_PATH to path where Patch is located, "
  "usually found in GIT_PATH/usr/bin on Windows")
endif()

include(${TOP_DIR}/cmake/external_libs/jpeg_turbo.cmake)

set(MINDDATA_DIR ${CCSRC_DIR}/minddata/dataset)
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -g2 -ggdb -fno-inline-functions -fno-omit-frame-pointer \
  -D_LIBCPP_INLINE_VISIBILITY='' -D_LIBCPP_DISABLE_EXTERN_TEMPLATE=1 -DHALF_ENABLE_CPP11_USER_LITERALS=0 \
  -D_FORTIFY_SOURCE=2 -Wno-cpp")
set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -Werror -Wno-return-std-move -Wno-unused-private-field \
  -Wno-unused-lambda-capture -Wno-sign-compare -Wno-overloaded-virtual -Wno-unneeded-internal-declaration \
  -Wno-unused-variable -Wno-pessimizing-move -Wno-inconsistent-missing-override")

set(CMAKE_CXX_FLAGS "$ENV{CXXFLAGS} -I/usr/local/include -std=c++17 -Wall -fPIC")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OPTION_CXX_FLAGS}")

if(PLATFORM_ARM)
    set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O2 -Werror -Wno-return-std-move -Wno-unused-private-field \
      -Wno-unused-lambda-capture -Wno-sign-compare -Wno-overloaded-virtual -Wno-unneeded-internal-declaration \
      -Wno-unused-variable -Wno-pessimizing-move -Wno-inconsistent-missing-override \
      -DHALF_ENABLE_CPP11_USER_LITERALS=0 -D_FORTIFY_SOURCE=2")
else()
    set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O2 -Wl,--allow-shlib-undefined -DHALF_ENABLE_CPP11_USER_LITERALS=0 \
      -D_FORTIFY_SOURCE=2")
endif()

set(CMAKE_C_FLAGS "-fPIC -fPIE -D_FORTIFY_SOURCE=2 -O2 -Wall -Werror -fstack-protector-strong -Wno-attributes \
  -Wno-deprecated-declarations -Wno-missing-braces ${CMAKE_C_FLAGS}")
set(CMAKE_CXX_FLAGS "-fPIC -fPIE -D_FORTIFY_SOURCE=2 -O2 -Wall -Werror -fstack-protector-strong -Wno-attributes \
  -Wno-deprecated-declarations -Wno-missing-braces -Wno-overloaded-virtual ${CMAKE_CXX_FLAGS}")

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
AUX_SOURCE_DIRECTORY(${MINDDATA_DIR}/engine/datasetops/source/sampler
  MINDDATA_ENGINE_DATASETOPS_SOURCE_SAMPLER_SRC_FILES)
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

AUX_SOURCE_DIRECTORY(${MINDDATA_DIR}/kernels/ir MINDDATA_KERNELS_IR_SRC_FILES)

if(BUILD_MINDDATA STREQUAL "full")
    include_directories("${MINDDATA_DIR}/kernels/image")
    include_directories("${MINDDATA_DIR}/liteapi")
    include_directories("${TOP_DIR}")

    set(MINDDATA_FULL_SRC
        ${TOP_DIR}/mindspore/lite/src/cxx_api/types.cc
        ${TOP_DIR}/mindspore/lite/src/cxx_api/tensor/tensor_impl.cc
        ${TOP_DIR}/mindspore/lite/src/tensor.cc
        ${TOP_DIR}/mindspore/lite/src/common/string_util.cc
        ${CORE_DIR}/utils/status.cc
        ${MINDDATA_DIR}/api/datasets.cc
        ${MINDDATA_DIR}/kernels/data/data_utils.cc
        ${MINDDATA_DIR}/api/samplers.cc
        ${MINDDATA_DIR}/api/iterator.cc
        ${MINDDATA_DIR}/api/data_helper.cc
        ${MINDDATA_DIR}/api/execute.cc
        ${MINDDATA_DIR}/core/de_tensor.cc
        ${MINDDATA_DIR}/core/tensor_shape.cc
        ${MINDDATA_DIR}/util/memory_pool.cc
        ${MINDDATA_DIR}/core/config_manager.cc
        ${MINDDATA_DIR}/core/data_type.cc
        ${MINDDATA_DIR}/core/tensor_helpers.cc
        ${MINDDATA_DIR}/core/tensor.cc
        ${MINDDATA_DIR}/core/global_context.cc
        ${MINDDATA_DIR}/core/client.cc
        ${MINDDATA_DIR}/engine/tree_adapter_lite.cc
        ${MINDDATA_DIR}/engine/consumers/pull_based_tree_consumer.cc
        ${MINDDATA_DIR}/engine/consumers/tree_consumer.cc
        ${MINDDATA_DIR}/engine/ir/datasetops/dataset_node.cc
        ${MINDDATA_DIR}/engine/ir/datasetops/epoch_ctrl_node.cc
        ${MINDDATA_DIR}/engine/ir/datasetops/batch_node.cc
        ${MINDDATA_DIR}/engine/ir/datasetops/map_node.cc
        ${MINDDATA_DIR}/engine/ir/datasetops/root_node.cc
        ${MINDDATA_DIR}/engine/ir/datasetops/repeat_node.cc
        ${MINDDATA_DIR}/engine/ir/datasetops/project_node.cc
        ${MINDDATA_DIR}/engine/ir/datasetops/shuffle_node.cc
        ${MINDDATA_DIR}/engine/ir/datasetops/source/album_node.cc
        ${MINDDATA_DIR}/engine/ir/datasetops/source/mnist_node.cc
        ${MINDDATA_DIR}/engine/ir/datasetops/source/samplers/samplers_ir.cc
        ${MINDDATA_DIR}/engine/datasetops/dataset_op.cc
        ${MINDDATA_DIR}/engine/datasetops/repeat_op.cc
        ${MINDDATA_DIR}/engine/datasetops/epoch_ctrl_op.cc
        ${MINDDATA_DIR}/engine/datasetops/device_queue_op.cc
        ${MINDDATA_DIR}/engine/datasetops/project_op.cc
        ${MINDDATA_DIR}/engine/datasetops/shuffle_op.cc
        ${MINDDATA_DIR}/engine/datasetops/pipeline_op.cc
        ${MINDDATA_DIR}/engine/datasetops/batch_op.cc
        ${MINDDATA_DIR}/engine/datasetops/parallel_op.cc
        ${MINDDATA_DIR}/engine/datasetops/map_op/map_op.cc
        ${MINDDATA_DIR}/engine/datasetops/map_op/cpu_map_job.cc
        ${MINDDATA_DIR}/engine/datasetops/source/album_op.cc
        ${MINDDATA_DIR}/engine/datasetops/source/mnist_op.cc

        ${MINDDATA_DIR}/engine/datasetops/source/io_block.cc
        ${MINDDATA_DIR}/engine/opt/pre/getter_pass.cc
        ${MINDDATA_DIR}/engine/opt/pre/input_validation_pass.cc
        ${MINDDATA_DIR}/engine/opt/pre/cache_validation_pass.cc
        ${MINDDATA_DIR}/engine/opt/pre/node_removal_pass.cc
        ${MINDDATA_DIR}/engine/opt/pre/epoch_ctrl_pass.cc
        ${MINDDATA_DIR}/engine/opt/pre/deep_copy_pass.cc
        ${MINDDATA_DIR}/engine/opt/post/auto_worker_pass.cc
        ${MINDDATA_DIR}/engine/opt/pass.cc
        ${MINDDATA_DIR}/engine/perf/profiling.cc
        ${MINDDATA_DIR}/engine/perf/monitor.cc
        ${MINDDATA_DIR}/engine/perf/device_queue_tracing.cc
        ${MINDDATA_DIR}/engine/perf/connector_size.cc
        ${MINDDATA_DIR}/engine/perf/connector_throughput.cc
        ${MINDDATA_DIR}/engine/perf/dataset_iterator_tracing.cc
        ${MINDDATA_DIR}/engine/datasetops/source/sampler/sampler.cc
        ${MINDDATA_DIR}/engine/datasetops/source/sampler/subset_sampler.cc
        ${MINDDATA_DIR}/engine/datasetops/source/sampler/distributed_sampler.cc
        ${MINDDATA_DIR}/engine/datasetops/source/sampler/pk_sampler.cc
        ${MINDDATA_DIR}/engine/datasetops/source/sampler/random_sampler.cc
        ${MINDDATA_DIR}/engine/datasetops/source/sampler/sequential_sampler.cc
        ${MINDDATA_DIR}/engine/datasetops/source/sampler/subset_random_sampler.cc
        ${MINDDATA_DIR}/engine/datasetops/source/sampler/weighted_random_sampler.cc
        ${MINDDATA_DIR}/engine/runtime_context.cc
        ${MINDDATA_DIR}/engine/tree_adapter.cc
        ${MINDDATA_DIR}/engine/data_buffer.cc
        ${MINDDATA_DIR}/engine/execution_tree.cc
        ${MINDDATA_DIR}/engine/dataset_iterator.cc
        ${MINDDATA_DIR}/core/tensor_row.cc
        ${MINDDATA_DIR}/api/vision.cc
        ${MINDDATA_DIR}/api/transforms.cc
        ${MINDDATA_DIR}/util/path.cc
        ${MINDDATA_DIR}/util/status.cc
        ${MINDDATA_DIR}/util/service.cc
        ${MINDDATA_DIR}/util/json_helper.cc
        ${MINDDATA_DIR}/util/cond_var.cc
        ${MINDDATA_DIR}/engine/data_schema.cc
        ${MINDDATA_DIR}/kernels/tensor_op.cc
        ${MINDDATA_DIR}/kernels/image/affine_op.cc
        ${MINDDATA_DIR}/kernels/image/lite_image_utils.cc
        ${MINDDATA_DIR}/kernels/image/center_crop_op.cc
        ${MINDDATA_DIR}/kernels/image/crop_op.cc
        ${MINDDATA_DIR}/kernels/image/decode_op.cc
        ${MINDDATA_DIR}/kernels/image/normalize_op.cc
        ${MINDDATA_DIR}/kernels/image/resize_op.cc
        ${MINDDATA_DIR}/kernels/image/resize_preserve_ar_op.cc
        ${MINDDATA_DIR}/kernels/image/rgb_to_gray_op.cc
        ${MINDDATA_DIR}/kernels/image/rotate_op.cc
        ${MINDDATA_DIR}/kernels/image/random_affine_op.cc
        ${MINDDATA_DIR}/kernels/image/math_utils.cc
        ${MINDDATA_DIR}/kernels/data/compose_op.cc
        ${MINDDATA_DIR}/kernels/data/duplicate_op.cc
        ${MINDDATA_DIR}/kernels/data/one_hot_op.cc
        ${MINDDATA_DIR}/kernels/data/random_apply_op.cc
        ${MINDDATA_DIR}/kernels/data/random_choice_op.cc
        ${MINDDATA_DIR}/kernels/data/type_cast_op.cc
        ${MINDDATA_DIR}/kernels/image/exif_utils.cc
        ${MINDDATA_DIR}/kernels/ir/validators.cc
        ${MINDDATA_DIR}/kernels/ir/data/transforms_ir.cc
        ${MINDDATA_DIR}/kernels/ir/vision/vision_ir.cc
        ${MINDDATA_DIR}/callback/callback_manager.cc
        ${MINDDATA_DIR}/util/task_manager.cc
        ${MINDDATA_DIR}/util/services.cc
        ${MINDDATA_DIR}/util/wait_post.cc
        ${MINDDATA_DIR}/util/task.cc
        ${MINDDATA_DIR}/util/circular_pool.cc
        ${MINDDATA_DIR}/util/lock.cc
        ${MINDDATA_DIR}/util/wait_post.cc
        ${MINDDATA_DIR}/util/intrp_service.cc
        ${MINDDATA_DIR}/util/arena.cc
        )

    add_library(minddata-lite SHARED
    ${MINDDATA_KERNELS_IMAGE_LITE_CV_FILES}
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/common/log_adapter.cc
    ${CORE_DIR}/utils/ms_utils.cc
    ${MINDDATA_FULL_SRC}
    )

    add_dependencies(minddata-lite fbs_src)

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
    elseif()
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
            ${MINDDATA_DIR}/core/de_tensor.cc
            ${MINDDATA_DIR}/api/vision.cc
            ${MINDDATA_DIR}/api/data_helper.cc
            ${MINDDATA_DIR}/api/execute.cc
            ${MINDDATA_DIR}/api/transforms.cc
            ${MINDDATA_DIR}/util/path.cc
            ${MINDDATA_DIR}/util/status.cc
            ${MINDDATA_DIR}/util/json_helper.cc
            ${MINDDATA_DIR}/util/memory_pool.cc
            ${MINDDATA_DIR}/engine/data_schema.cc
            ${MINDDATA_DIR}/kernels/tensor_op.cc
            ${MINDDATA_DIR}/kernels/image/lite_image_utils.cc
            ${MINDDATA_DIR}/kernels/image/center_crop_op.cc
            ${MINDDATA_DIR}/kernels/image/crop_op.cc
            ${MINDDATA_DIR}/kernels/image/normalize_op.cc
            ${MINDDATA_DIR}/kernels/image/resize_op.cc
            ${MINDDATA_DIR}/kernels/image/resize_preserve_ar_op.cc
            ${MINDDATA_DIR}/kernels/image/rgb_to_gray_op.cc
            ${MINDDATA_DIR}/kernels/image/rotate_op.cc
            ${MINDDATA_DIR}/kernels/data/compose_op.cc
            ${MINDDATA_DIR}/kernels/data/duplicate_op.cc
            ${MINDDATA_DIR}/kernels/data/one_hot_op.cc
            ${MINDDATA_DIR}/kernels/data/random_apply_op.cc
            ${MINDDATA_DIR}/kernels/data/random_choice_op.cc
            ${MINDDATA_DIR}/kernels/data/type_cast_op.cc
            ${MINDDATA_DIR}/kernels/data/data_utils.cc
            ${MINDDATA_DIR}/kernels/image/exif_utils.cc
            ${MINDDATA_DIR}/kernels/ir/validators.cc
            ${MINDDATA_DIR}/kernels/ir/data/transforms_ir.cc
            ${MINDDATA_DIR}/kernels/ir/vision/vision_ir.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/wrapper/MDToDApi.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/wrapper/album_op_android.cc
            )

    set(MINDSPORE_LITE_CXXAPI_SRC
            ${CORE_DIR}/utils/status.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/../src/cxx_api/types.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/../src/tensor.cc
            )

    add_library(minddata-lite SHARED
            ${MINDDATA_KERNELS_IMAGE_LITE_CV_FILES}
            ${CMAKE_CURRENT_SOURCE_DIR}/../src/common/log_adapter.cc
            ${CORE_DIR}/utils/ms_utils.cc
            ${MINDDATA_TODAPI_SRC}
            ${MINDSPORE_LITE_CXXAPI_SRC}
            )

    add_dependencies(minddata-lite fbs_src)

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
        ${MINDDATA_DIR}/core/de_tensor.cc
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
        "${MINDDATA_DIR}/kernels/image/resize_preserve_ar_op.cc"
        "${MINDDATA_DIR}/kernels/image/rgb_to_gray_op.cc"
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
        ${MINDDATA_DIR}/kernels/ir/validators.cc
        ${MINDDATA_DIR}/kernels/ir/data/transforms_ir.cc
        ${MINDDATA_DIR}/kernels/ir/vision/vision_ir.cc
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
