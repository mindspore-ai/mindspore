set(MINDDATA_DIR ${CCSRC_DIR}/minddata/dataset)
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -g2 -ggdb -fno-inline-functions -fno-omit-frame-pointer -D_LIBCPP_INLINE_VISIBILITY='' -D_LIBCPP_DISABLE_EXTERN_TEMPLATE=1 -DHALF_ENABLE_CPP11_USER_LITERALS=0 -D_FORTIFY_SOURCE=2 -Wno-cpp")

set(CMAKE_CXX_FLAGS "$ENV{CXXFLAGS} -I/usr/local/include -std=c++17 -Wall -fPIC")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OPTION_CXX_FLAGS}")

if (PLATFORM_ARM64 OR PLATFORM_ARM32)
    set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O2 -Werror -Wno-return-std-move -Wno-unused-private-field -Wno-unused-lambda-capture -Wno-sign-compare -Wno-overloaded-virtual -Wno-unneeded-internal-declaration -Wno-unused-variable -Wno-pessimizing-move -Wno-inconsistent-missing-override -DHALF_ENABLE_CPP11_USER_LITERALS=0 -D_FORTIFY_SOURCE=2")
else()
    set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O2 -Wl,--allow-shlib-undefined -DHALF_ENABLE_CPP11_USER_LITERALS=0 -D_FORTIFY_SOURCE=2")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-non-virtual-dtor")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-sign-compare")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-private-field")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-reorder")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-switch")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-sequence-point")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-variable")
if (PLATFORM_ARM64 OR PLATFORM_ARM32)
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

AUX_SOURCE_DIRECTORY(${MINDDATA_DIR}/engine/datasetops MINDDATA_ENGINE_DATASETOPS_SRC_FILES)
AUX_SOURCE_DIRECTORY(${MINDDATA_DIR}/engine/datasetops/map_op MINDDATA_ENGINE_DATASETOPS_MAPOP_SRC_FILES)
AUX_SOURCE_DIRECTORY(${MINDDATA_DIR}/engine/datasetops/source MINDDATA_ENGINE_DATASETOPS_SOURCE_SRC_FILES)
AUX_SOURCE_DIRECTORY(${MINDDATA_DIR}/engine/datasetops/source/sampler MINDDATA_ENGINE_DATASETOPS_SOURCE_SAMPLER_SRC_FILES)

AUX_SOURCE_DIRECTORY(${MINDDATA_DIR}/engine/opt MINDDATA_ENGINE_OPT_SRC_FILES)
AUX_SOURCE_DIRECTORY(${MINDDATA_DIR}/engine/opt/optional MINDDATA_ENGINE_OPT_OPTIONAL_SRC_FILES)
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

if (BUILD_MINDDATA STREQUAL "full")
    list(REMOVE_ITEM MINDDATA_API_SRC_FILES 
        "${MINDDATA_DIR}/api/text.cc" 
        "${MINDDATA_DIR}/api/de_tensor.cc"
        "${MINDDATA_DIR}/api/execute.cc"
        )
    
    list(REMOVE_ITEM MINDDATA_CALLBACK_SRC_FILES 
        "${MINDDATA_DIR}/callback/py_ds_callback.cc" 
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
        )

    list(REMOVE_ITEM MINDDATA_ENGINE_DATASETOPS_SOURCE_SRC_FILES 
        "${MINDDATA_DIR}/engine/datasetops/source/generator_op.cc"
        "${MINDDATA_DIR}/engine/datasetops/source/voc_op.cc"
        "${MINDDATA_DIR}/engine/datasetops/source/manifest_op.cc"
        "${MINDDATA_DIR}/engine/datasetops/source/mindrecord_op.cc"
        "${MINDDATA_DIR}/engine/datasetops/source/tf_reader_op.cc"
        )

    list(REMOVE_ITEM MINDDATA_ENGINE_DATASETOPS_SOURCE_SAMPLER_SRC_FILES
        "${MINDDATA_DIR}/engine/datasetops/source/sampler/python_sampler.cc"
        )

        list(REMOVE_ITEM MINDDATA_ENGINE_OPT_POST_SRC_FILES
        "${MINDDATA_DIR}/engine/opt/post/repeat_pass.cc"
        )

    list(REMOVE_ITEM MINDDATA_ENGINE_OPT_PRE_SRC_FILES
        "${MINDDATA_DIR}/engine/opt/pre/cache_transform_pass.cc"
        )

    include_directories("${CMAKE_BINARY_DIR}/minddata/dataset/engine/cache")

    if (BUILD_MINDDATA_EXAMPLE AND (PLATFORM_ARM32 OR PLATFORM_ARM64))
        set(MINDDATA_EXAMPLE_SRC ${CMAKE_CURRENT_SOURCE_DIR}/example/jni-example.cc)
    endif()

    add_library(minddata-lite SHARED
        ${MINDDATA_API_SRC_FILES}
        ${MINDDATA_CALLBACK_SRC_FILES}
        ${MINDDATA_CORE_SRC_FILES}
        ${MINDDATA_ENGINE_SRC_FILES}
        #${MINDDATA_ENGINE_CACHE_SRC_FILES}
        ${MINDDATA_ENGINE_DATASETOPS_SRC_FILES}
        ${MINDDATA_ENGINE_DATASETOPS_MAPOP_SRC_FILES}
        ${MINDDATA_ENGINE_DATASETOPS_SOURCE_SRC_FILES}
        ${MINDDATA_ENGINE_DATASETOPS_SOURCE_SAMPLER_SRC_FILES}
        ${MINDDATA_ENGINE_OPT_SRC_FILES}
        ${MINDDATA_ENGINE_OPT_OPTIONAL_SRC_FILES}
        ${MINDDATA_ENGINE_OPT_POST_SRC_FILES}
        ${MINDDATA_ENGINE_OPT_PRE_SRC_FILES}
        ${MINDDATA_ENGINE_OPT_UTIL_SRC_FILES}
        ${MINDDATA_ENGINE_PERF_SRC_FILES}
        ${MINDDATA_KERNELS_SRC_FILES}
        ${MINDDATA_KERNELS_IMAGE_SRC_FILES}
        ${MINDDATA_KERNELS_DATA_SRC_FILES}
        ${MINDDATA_UTIL_SRC_FILES}
        ${MINDDATA_EXAMPLE_SRC}
        ${CMAKE_CURRENT_SOURCE_DIR}/../src/common/log_adapter.cc
        ${CORE_DIR}/utils/ms_utils.cc
        ${CORE_DIR}/gvar/logging_level.cc
        ${CCSRC_DIR}/utils/system/crc32c.cc
        )

    target_link_libraries(minddata-lite
        securec
        jpeg-turbo
        jpeg
        opencv_core
        opencv_imgcodecs
        opencv_imgproc
        mindspore::json
        )
    
    # ref: https://github.com/android/ndk/issues/1202
    if (PLATFORM_ARM32)
        file(GLOB_RECURSE LIBCLANG_RT_LIB $ENV{ANDROID_NDK}/libclang_rt.builtins-arm-android.a)
        if (LIBCLANG_RT_LIB STREQUAL "")
            MESSAGE(FATAL_ERROR "Cannot find libclang_rt.builtins-arm-androi2d.a in $ENV{ANDROID_NDK}")
        endif()
        target_link_libraries(minddata-lite ${LIBCLANG_RT_LIB})
    endif()

    if (PLATFORM_ARM32 OR PLATFORM_ARM64)
        target_link_libraries(minddata-lite log)
    elseif (BUILD_MINDDATA_EXAMPLE)
        add_executable(mdlite-example ${CMAKE_CURRENT_SOURCE_DIR}/example/x86-example.cc)
        target_link_libraries(mdlite-example minddata-lite)
        add_custom_command(TARGET mdlite-example POST_BUILD
            COMMAND cp -rf ${CMAKE_CURRENT_SOURCE_DIR}/example/testCifar10Data ${CMAKE_BINARY_DIR}/minddata
            )
    endif()
    
elseif (BUILD_MINDDATA STREQUAL "lite")
    list(REMOVE_ITEM MINDDATA_CORE_SRC_FILES "${MINDDATA_DIR}/core/client.cc")
    list(REMOVE_ITEM MINDDATA_KERNELS_SRC_FILES "${MINDDATA_DIR}/kernels/py_func_op.cc")
    add_library(minddata-eager OBJECT 
        ${MINDDATA_DIR}/api/de_tensor.cc
        ${MINDDATA_DIR}/api/execute.cc
        )

    add_library(minddata-lite SHARED
        ${MINDDATA_CORE_SRC_FILES}
        ${MINDDATA_KERNELS_SRC_FILES}
        ${MINDDATA_KERNELS_IMAGE_SRC_FILES}
        ${MINDDATA_KERNELS_DATA_SRC_FILES}
        ${MINDDATA_DIR}/util/status.cc
        ${MINDDATA_DIR}/util/memory_pool.cc
        ${MINDDATA_DIR}/util/path.cc
        ${MINDDATA_DIR}/api/transforms.cc
        ${CORE_DIR}/utils/log_adapter.cc
        ${CORE_DIR}/gvar/logging_level.cc
        )

    target_link_libraries(minddata-lite
        securec
        jpeg-turbo
        jpeg
        opencv_core
        opencv_imgcodecs
        opencv_imgproc
        mindspore::json
        )

    # ref: https://github.com/android/ndk/issues/1202
    if (PLATFORM_ARM32)
        file(GLOB_RECURSE LIBCLANG_RT_LIB $ENV{ANDROID_NDK}/libclang_rt.builtins-arm-android.a)
        if (LIBCLANG_RT_LIB STREQUAL "")
            MESSAGE(FATAL_ERROR "Cannot find libclang_rt.builtins-arm-androi2d.a in $ENV{ANDROID_NDK}")
        endif()
        target_link_libraries(minddata-lite ${LIBCLANG_RT_LIB})
    endif()
endif()
