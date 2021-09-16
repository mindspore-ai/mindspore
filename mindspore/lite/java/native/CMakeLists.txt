cmake_minimum_required(VERSION 3.10)
project(Lite-java)

set(BUILD_LITE "on")

include(${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/secure_option.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/compile_link_option.cmake)

if(TARGET_HIMIX200)
    set(CMAKE_CXX_FLAGS "-Wno-error=maybe-uninitialized ${CMAKE_CXX_FLAGS}")
endif()

if(PLATFORM_ARM64 OR PLATFORM_ARM32)
    set(PLATFORM_ARM "on")
    add_compile_definitions(PLATFORM_ARM)
endif()

if(PLATFORM_ARM32 OR PLATFORM_ARM64)
    #for performance
    if("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -fomit-frame-pointer -fstrict-aliasing -ffunction-sections \
        -fdata-sections -ffast-math -fno-rtti -fno-exceptions")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -fomit-frame-pointer -fstrict-aliasing -ffunction-sections \
        -fdata-sections -ffast-math -fno-rtti -fno-exceptions")
        if(TARGET_HIMIX200)
            string(REPLACE "-fno-rtti " "" CMAKE_C_FLAGS ${CMAKE_C_FLAGS})
        endif()
    endif()
endif()

set(MS_VERSION_MAJOR ${MS_VERSION_MAJOR})
set(MS_VERSION_MINOR ${MS_VERSION_MINOR})
set(MS_VERSION_REVISION ${MS_VERSION_REVISION})
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DMS_VERSION_MAJOR=${MS_VERSION_MAJOR} -DMS_VERSION_MINOR=${MS_VERSION_MINOR} \
  -DMS_VERSION_REVISION=${MS_VERSION_REVISION}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DMS_VERSION_MAJOR=${MS_VERSION_MAJOR} -DMS_VERSION_MINOR=${MS_VERSION_MINOR} \
  -DMS_VERSION_REVISION=${MS_VERSION_REVISION}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")

#set for cross-compiling toolchain
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY BOTH)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE BOTH)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE BOTH)

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

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${LITE_DIR}) ## lite include
include_directories(${TOP_DIR}) ## api include
include_directories(${TOP_DIR}/mindspore/core/) ## core include
include_directories(${LITE_DIR}/build) ## flatbuffers
if(PLATFORM_ARM64 OR PLATFORM_ARM32)
    link_directories(${CMAKE_CURRENT_SOURCE_DIR}/libs/${ANDROID_ABI}/)
else()
    link_directories(${CMAKE_CURRENT_SOURCE_DIR}/libs/linux_x86/)
    set(JAVA_HOME "$ENV{JAVA_HOME}")
    include_directories(${JAVA_HOME}/include/) ## jni
    include_directories(${JAVA_HOME}/include/linux) ## jni
endif()
set(JNI_SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/runtime/model.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/runtime/version.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/runtime/ms_config.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/runtime/ms_tensor.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/runtime/lite_session.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/common/jni_utils.cpp
  )

set(LITE_SO_NAME mindspore-lite)

add_library(mindspore-lite-jni SHARED ${JNI_SRC})

if(PLATFORM_ARM64 OR PLATFORM_ARM32)
    find_library(log-lib log)
    target_link_libraries(mindspore-lite-jni ${LITE_SO_NAME} ${log-lib})
else()
    target_link_libraries(mindspore-lite-jni ${LITE_SO_NAME})
endif()

if(SUPPORT_TRAIN)
    set(LITE_TRAIN_SO_NAME mindspore-lite-train minddata-lite)
    set(JNI_TRAIN_SRC ${CMAKE_CURRENT_SOURCE_DIR}/runtime/train_session.cpp)
    add_library(mindspore-lite-train-jni SHARED ${JNI_TRAIN_SRC})
    if(PLATFORM_ARM64 OR PLATFORM_ARM32)
        find_library(log-lib log)
        target_link_libraries(mindspore-lite-train-jni ${LITE_TRAIN_SO_NAME} ${log-lib})
    else()
        target_link_libraries(mindspore-lite-train-jni ${LITE_TRAIN_SO_NAME})
    endif()
endif()
