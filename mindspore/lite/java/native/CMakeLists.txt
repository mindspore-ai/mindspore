cmake_minimum_required(VERSION 3.10)
project(Lite-java)

set(BUILD_LITE "on")

if(PLATFORM_ARM32 OR PLATFORM_ARM32)
    set(PLATFORM_ARM "on")
    add_compile_definitions(PLATFORM_ARM)
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
        ${CMAKE_CURRENT_SOURCE_DIR}/common/jni_utils.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/runtime/model.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/runtime/version.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/runtime/ms_config.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/runtime/ms_tensor.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/runtime/lite_session.cpp
  )

if(SUPPORT_TRAIN)
  set(JNI_SRC
  ${JNI_SRC}
  ${CMAKE_CURRENT_SOURCE_DIR}/runtime/train_session.cpp
  )
endif()

add_library(mindspore-lite-jni SHARED ${JNI_SRC})

if(PLATFORM_ARM64 OR PLATFORM_ARM32)
    find_library(log-lib log)
    target_link_libraries(mindspore-lite-jni mindspore-lite ${log-lib})
else()
    target_link_libraries(mindspore-lite-jni mindspore-lite)
endif()
