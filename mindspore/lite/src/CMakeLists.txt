add_compile_definitions(USE_ANDROID_LOG)
set(LITE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/..)
include_directories(${LITE_DIR}/nnacl/)
include_directories(${LITE_DIR}/nnacl/optimize)

if (PLATFORM_ARM32 OR PLATFORM_ARM64)
    #for performance
    if ("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
        if (SUPPORT_GPU)
            set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -fomit-frame-pointer -fstrict-aliasing -ffunction-sections -fdata-sections -ffast-math -fno-rtti")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -fomit-frame-pointer -fstrict-aliasing -ffunction-sections -fdata-sections -ffast-math -fno-rtti")
        else ()
            set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -fomit-frame-pointer -fstrict-aliasing -ffunction-sections -fdata-sections -ffast-math -fno-rtti -fno-exceptions")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -fomit-frame-pointer -fstrict-aliasing -ffunction-sections -fdata-sections -ffast-math -fno-rtti -fno-exceptions")
        endif ()
    endif ()
endif ()

set(LITE_SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/common/graph_util.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/common/log_adapter.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/common/string_util.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/runtime/allocator.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/runtime/runtime_api.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/runtime/thread_pool.c
        ${CMAKE_CURRENT_SOURCE_DIR}/runtime/workspace_pool.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/tensor.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/executor.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/inner_context.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/kernel_registry.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/lite_kernel.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/populate_parameter.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/scheduler.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/lite_session.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/model.cc
        )

if (SUPPORT_GPU)
    set(LITE_SRC
            ${LITE_SRC}
            ${CMAKE_CURRENT_SOURCE_DIR}/runtime/kernel/opencl/subgraph_opencl_kernel.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/runtime/kernel/opencl/utils.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/runtime/opencl/opencl_executor.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/runtime/opencl/opencl_allocator.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/runtime/opencl/opencl_runtime.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/runtime/opencl/opencl_wrapper.cc
            )
endif ()


if (SUPPORT_TRAIN)
    set(ANF_SRC
            ${ANF_SRC}
            )
    set(PASS_SRC)
    set(LITE_SRC
            ${LITE_SRC}
            ${ANF_SRC}
            ${CMAKE_CURRENT_SOURCE_DIR}/train/train_populate_parameter.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/train/train_session.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/train/train_model.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/lite_session.cc
            )
endif ()

file(GLOB_RECURSE C_OPS_SRC ${CMAKE_CURRENT_SOURCE_DIR}/ops/*.cc)
add_subdirectory(runtime/kernel/arm)

add_library(mindspore-lite SHARED ${LITE_SRC} ${C_OPS_SRC})
add_dependencies(mindspore-lite fbs_src)
set_target_properties(mindspore-lite PROPERTIES CLEAN_DIRECT_OUTPUT 1)

add_library(mindspore-lite_static STATIC ${LITE_SRC} ${C_OPS_SRC})
add_dependencies(mindspore-lite_static fbs_src)
set_target_properties(mindspore-lite_static PROPERTIES OUTPUT_NAME "mindspore-lite")
set_target_properties(mindspore-lite_static PROPERTIES CLEAN_DIRECT_OUTPUT 1)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-private-field")
if (SUPPORT_GPU)
    add_subdirectory(runtime/kernel/opencl)
    target_link_libraries(mindspore-lite cpu_kernel_mid opencl_kernel_mid nnacl)
    target_link_libraries(mindspore-lite_static cpu_kernel_mid opencl_kernel_mid nnacl_mid)
else ()
    target_link_libraries(mindspore-lite cpu_kernel_mid nnacl)
    target_link_libraries(mindspore-lite_static cpu_kernel_mid nnacl_mid)
endif ()
if (PLATFORM_ARM32 OR PLATFORM_ARM64)
    target_link_libraries(mindspore-lite log)
    target_link_libraries(mindspore-lite_static log)
endif ()
if (BUILD_MINDDATA STREQUAL "lite")
    target_link_libraries(mindspore-lite minddata_eager_mid minddata-lite)
    target_link_libraries(mindspore-lite_static minddata_eager_mid)
endif ()

if ("${CMAKE_BUILD_TYPE}" STREQUAL "Release" AND PLATFORM_ARM)
    add_custom_command(TARGET mindspore-lite POST_BUILD
            COMMAND ${ANDROID_NDK}/toolchains/aarch64-linux-android-4.9/prebuilt/linux-x86_64/aarch64-linux-android/bin/strip
            ${TOP_DIR}/mindspore/lite/build/src/libmindspore-lite.so)
endif ()

if ("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
    if (PLATFORM_ARM)
        add_custom_command(TARGET mindspore-lite POST_BUILD
                COMMAND ${ANDROID_NDK}/toolchains/aarch64-linux-android-4.9/prebuilt/linux-x86_64/aarch64-linux-android/bin/strip
                ${CMAKE_BINARY_DIR}/src/libmindspore-lite.so)
    elseif (NOT WIN32)
        add_custom_command(TARGET mindspore-lite POST_BUILD
                COMMAND strip ${CMAKE_BINARY_DIR}/src/libmindspore-lite.so)
    endif ()
endif ()

########################## build optimize and float16 library #################################3
if (PLATFORM_ARM64)
    add_library(mindspore-lite-optimize SHARED)
    target_link_libraries(mindspore-lite-optimize cpu_opt_kernel_mid)
    target_link_libraries(mindspore-lite-optimize nnacl_optimize)

    add_library(mindspore-lite-fp16 SHARED)
    target_link_libraries(mindspore-lite-fp16 cpu_fp16_kernel_mid)
    target_link_libraries(mindspore-lite-fp16 nnacl_fp16)
endif ()

if ("${CMAKE_BUILD_TYPE}" STREQUAL "Release" AND (PLATFORM_ARM64))
    add_custom_command(TARGET mindspore-lite-optimize POST_BUILD COMMAND
            ${ANDROID_NDK}/toolchains/aarch64-linux-android-4.9/prebuilt/linux-x86_64/aarch64-linux-android/bin/strip
            ${TOP_DIR}/mindspore/lite/build/src/libmindspore-lite-optimize.so)

    add_custom_command(TARGET mindspore-lite-fp16 POST_BUILD COMMAND
            ${ANDROID_NDK}/toolchains/aarch64-linux-android-4.9/prebuilt/linux-x86_64/aarch64-linux-android/bin/strip
            ${TOP_DIR}/mindspore/lite/build/src/libmindspore-lite-fp16.so)
endif ()

