cmake_minimum_required(VERSION 3.10)
project(Lite-java)

set(VERSION_STR "1.7.0" CACHE STRING "get from version")
add_definitions(-DVERSION_STR=\"${VERSION_STR}\")

set(BUILD_LITE "on")

include(${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/secure_option.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/compile_link_option.cmake)

if(PLATFORM_ARM64 OR PLATFORM_ARM32)
    set(PLATFORM_ARM "on")
    add_compile_definitions(PLATFORM_ARM)
endif()

if(CMAKE_BUILD_TYPE)
    if("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
    message("build release")
        set(CMAKE_CXX_VISIBILITY_PRESET hidden)
        set(CMAKE_C_VISIBILITY_PRESET hidden)
    endif()
endif()

if(PLATFORM_ARM32 OR PLATFORM_ARM64)
    #for performance
    if("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -fomit-frame-pointer -fstrict-aliasing -ffunction-sections \
        -fdata-sections -ffast-math -fno-rtti -fno-exceptions")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -fomit-frame-pointer -fstrict-aliasing -ffunction-sections \
        -fdata-sections -ffast-math -fno-rtti -fno-exceptions")
    endif()
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
set(CMAKE_SKIP_BUILD_RPATH on)

#set for cross-compiling toolchain
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY BOTH)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE BOTH)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE BOTH)

if(DEFINED ENV{MSLITE_ENABLE_RUNTIME_GLOG})
    set(MSLITE_ENABLE_RUNTIME_GLOG $ENV{MSLITE_ENABLE_RUNTIME_GLOG})
endif()

if(DEFINED ENV{MSLITE_ENABLE_SERVER_INFERENCE})
    set(MSLITE_ENABLE_SERVER_INFERENCE $ENV{MSLITE_ENABLE_SERVER_INFERENCE})
endif()

if(DEFINED ENV{MSLITE_ENABLE_PARALLEL_INFERENCE})
    set(MSLITE_ENABLE_PARALLEL_INFERENCE $ENV{MSLITE_ENABLE_PARALLEL_INFERENCE})
endif()

if(DEFINED ENV{MSLITE_ENABLE_CLOUD_INFERENCE})
    set(MSLITE_ENABLE_CLOUD_INFERENCE $ENV{MSLITE_ENABLE_CLOUD_INFERENCE})
endif()

if(DEFINED ENV{MSLITE_ENABLE_CLOUD_FUSION_INFERENCE})
    set(MSLITE_ENABLE_CLOUD_FUSION_INFERENCE $ENV{MSLITE_ENABLE_CLOUD_FUSION_INFERENCE})
endif()

if(DEFINED ENV{ENABLE_GLIBCXX})
    set(ENABLE_GLIBCXX $ENV{ENABLE_GLIBCXX})
endif()

if(MSLITE_ENABLE_RUNTIME_GLOG)
    set(USE_GLOG on)
endif()

if(MSLITE_ENABLE_SERVER_INFERENCE)
    set(MSLITE_ENABLE_PARALLEL_INFERENCE on)
    set(USE_GLOG on)
endif()

if(MSLITE_ENABLE_PARALLEL_INFERENCE)
    set(USE_GLOG on)
endif()

if(MSLITE_ENABLE_CLOUD_FUSION_INFERENCE OR MSLITE_ENABLE_CLOUD_INFERENCE)
    set(MSLITE_ENABLE_PARALLEL_INFERENCE on)
endif()

if(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    if(NOT ENABLE_GLIBCXX)
        add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)
    endif()
endif()

if(ENABLE_VERBOSE)
    set(CMAKE_VERBOSE_MAKEFILE on)
endif()

if(PLATFORM_ARM32)
    add_compile_definitions(ENABLE_ARM32)
endif()
if(PLATFORM_ARM64)
    add_compile_definitions(ENABLE_ARM64)
endif()

set(TOP_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../..)
set(LITE_DIR ${TOP_DIR}/mindspore/lite)
set(NEW_NATIVE_DIR ${LITE_DIR}/java/src/main/native)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../src)##for common/log_adapter.h
include_directories(${LITE_DIR}) ## lite include
include_directories(${TOP_DIR}) ## api include
include_directories(${TOP_DIR}/mindspore/core/) ## core include
include_directories(${LITE_DIR}/build) ## flatbuffers

if(PLATFORM_ARM64 OR PLATFORM_ARM32)
    set(PLATFORM_DIR ${CMAKE_CURRENT_SOURCE_DIR}/libs/${ANDROID_ABI}/)
elseif(MACHINE_LINUX_ARM64)
    set(PLATFORM_DIR ${CMAKE_CURRENT_SOURCE_DIR}/libs/linux_aarch64/)
else()
    set(PLATFORM_DIR ${CMAKE_CURRENT_SOURCE_DIR}/libs/linux_x86/)
endif()

link_directories(${PLATFORM_DIR})

set(JAVA_HOME "$ENV{JAVA_HOME}")
include_directories(${JAVA_HOME}/include/) ## jni
include_directories(${JAVA_HOME}/include/linux) ## jni

set(JNI_SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/common/jni_utils.cpp
        ${NEW_NATIVE_DIR}/graph.cpp
        ${NEW_NATIVE_DIR}/model.cpp
        ${NEW_NATIVE_DIR}/ms_context.cpp
        ${NEW_NATIVE_DIR}/ms_tensor.cpp
        ${NEW_NATIVE_DIR}/version.cpp
        )

if(MSLITE_ENABLE_PARALLEL_INFERENCE)
    set(JNI_SRC
            ${JNI_SRC}
            ${NEW_NATIVE_DIR}/runner_config.cpp
            ${NEW_NATIVE_DIR}/model_parallel_runner.cpp
            )
endif()

if(USE_GLOG)
    set_property(SOURCE ${JNI_SRC} PROPERTY COMPILE_DEFINITIONS
            LOG_HDR_FILE_REL_PATH="mindspore/lite/java/native/../../../../mindspore/core/utils/log_adapter.h")
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../../../third_party/securec/include)
    include_directories(${CMAKE_BINARY_DIR}/jni_include)
    add_definitions(-DUSE_GLOG)
    set_property(SOURCE ${JNI_SRC} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_LITE)
    set(JNI_SRC
            ${JNI_SRC}
            ${TOP_DIR}/mindspore/core/utils/log_adapter.cc
            )
else()
    set(JNI_SRC
            ${JNI_SRC}
            ${TOP_DIR}/mindspore/lite/src/common/log.cc
            )
    set_property(SOURCE ${JNI_SRC} PROPERTY COMPILE_DEFINITIONS
            LITE_LOG_HEAR_FILE_REL_PATH="mindspore/lite/java/native/../../src/common/log.h")
endif()

set(LITE_SO_NAME mindspore-lite)

add_library(mindspore-lite-jni SHARED ${JNI_SRC})

if(PLATFORM_ARM64 OR PLATFORM_ARM32)
    find_library(log-lib log)
    target_link_libraries(mindspore-lite-jni ${LITE_SO_NAME} ${log-lib})
else()
    target_link_libraries(mindspore-lite-jni ${LITE_SO_NAME})
    if(USE_GLOG)
        find_library(GLOG_LIB
        NAMES mindspore_glog libmindspore_glog.so.0
        PATHS ${PLATFORM_DIR}
        )
        if(GLOG_LIB)
            message("Found glog lib :${GLOG_LIB}")
        else()
            message(FATAL_ERROR "No glog lib found, exit.")
        endif()
        target_link_libraries(mindspore-lite-jni -Wl,--no-as-needed ${GLOG_LIB} -Wl,--as-needed)
    endif()
endif()

if(SUPPORT_TRAIN)
    set(LITE_TRAIN_SO_NAME mindspore-lite-train)
    set(JNI_TRAIN_SRC
            ${NEW_NATIVE_DIR}/train_config.cpp
            )

    if(USE_GLOG)
        set_property(SOURCE ${JNI_SRC} PROPERTY COMPILE_DEFINITIONS
                LOG_HDR_FILE_REL_PATH="mindspore/lite/java/native/../../../../mindspore/core/utils/log_adapter.h")
        include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../../../third_party/securec/include)
        include_directories(${CMAKE_BINARY_DIR}/jni_include)
        add_definitions(-DUSE_GLOG)
        set(JNI_TRAIN_SRC
                ${JNI_TRAIN_SRC}
                ${TOP_DIR}/mindspore/core/utils/log_adapter.cc
                )
        set_property(SOURCE ${JNI_TRAIN_SRC} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_LITE)
    else()
        set(JNI_TRAIN_SRC
                ${JNI_TRAIN_SRC}
                ${TOP_DIR}/mindspore/lite/src/common/log.cc
                )
        set_property(SOURCE ${JNI_TRAIN_SRC} PROPERTY COMPILE_DEFINITIONS
                LITE_LOG_HEAR_FILE_REL_PATH="mindspore/lite/java/native/../../src/common/log.h")
    endif()
    add_library(mindspore-lite-train-jni SHARED ${JNI_TRAIN_SRC})
    if(PLATFORM_ARM64 OR PLATFORM_ARM32)
        find_library(log-lib log)
        target_link_libraries(mindspore-lite-train-jni -Wl,--no-as-needed ${LITE_TRAIN_SO_NAME} -Wl,--as-needed
                ${log-lib})
    else()
        if(USE_GLOG)
            target_link_libraries(mindspore-lite-train-jni -Wl,--no-as-needed ${GLOG_LIB} -Wl,--as-needed)
        endif()
        target_link_libraries(mindspore-lite-train-jni -Wl,--no-as-needed ${LITE_TRAIN_SO_NAME} -Wl,--as-needed)
    endif()
endif()
