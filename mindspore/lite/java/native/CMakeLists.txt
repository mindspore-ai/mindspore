cmake_minimum_required(VERSION 3.10)
project(Lite-java)

set(BUILD_LITE "on")

set(CMAKE_SKIP_RPATH TRUE)
set(CMAKE_SKIP_BUILD_RPATH TRUE)
set(CMAKE_SKIP_INSTALL_RPATH TRUE)

if(PLATFORM_ARM64 OR PLATFORM_ARM32)
    set(PLATFORM_ARM "on")
    add_compile_definitions(PLATFORM_ARM)
endif()

if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DDebug -g")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DDebug -g")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fvisibility=default")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=default")
else()
    ## enable for binscope for release
    set(CMAKE_C_FLAGS "-fPIC -fPIE -D_FORTIFY_SOURCE=2 -O2 -Wall -Werror -fstack-protector-strong -Wno-attributes \
    -Wno-deprecated-declarations -Wno-missing-braces ${CMAKE_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "-fPIC -fPIE -D_FORTIFY_SOURCE=2 -O2 -Wall -Werror -fstack-protector-strong -Wno-attributes \
    -Wno-deprecated-declarations -Wno-missing-braces -Wno-overloaded-virtual ${CMAKE_CXX_FLAGS}")
    if(TARGET_HIMIX200)
        set(CMAKE_CXX_FLAGS "-Wno-error=maybe-uninitialized ${CMAKE_CXX_FLAGS}")
    endif()
    if(NOT WIN32)
        set(CMAKE_SHARED_LINKER_FLAGS "-Wl,-z,relro,-z,now -Wl,-z,noexecstack -s ${CMAKE_SHARED_LINKER_FLAGS}")
        set(CMAKE_EXE_LINKER_FLAGS "-Wl,-z,relro,-z,now -Wl,-z,noexecstack -s -pie ${CMAKE_EXE_LINKER_FLAGS}")
        if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
            string(REPLACE "-s " "" CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS}")
            string(REPLACE "-s " "" CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")
        endif()
    endif()
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

if(SUPPORT_TRAIN)
  set(JNI_SRC
          ${JNI_SRC}
          ${CMAKE_CURRENT_SOURCE_DIR}/runtime/train_session.cpp
  )
endif()
add_library(mindspore-lite-jni SHARED ${JNI_SRC})

if(PLATFORM_ARM64 OR PLATFORM_ARM32)
    find_library(log-lib log)
    target_link_libraries(mindspore-lite-jni ${LITE_SO_NAME} ${log-lib})
else()
    target_link_libraries(mindspore-lite-jni ${LITE_SO_NAME})
endif()

if(SUPPORT_TRAIN)
  set(LITE_TRAIN_SO_NAME mindspore-lite-train minddata-lite)
  if(PLATFORM_ARM64 OR PLATFORM_ARM32)
    find_library(log-lib log)
    target_link_libraries(mindspore-lite-jni ${LITE_TRAIN_SO_NAME} ${log-lib})
  else()
    target_link_libraries(mindspore-lite-jni ${LITE_TRAIN_SO_NAME})
  endif()
endif()

set(NDK_STRIP
        "${ANDROID_NDK}/toolchains/aarch64-linux-android-4.9/prebuilt/linux-x86_64/aarch64-linux-android/bin/strip")
if("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
    if(PLATFORM_ARM64 OR PLATFORM_ARM32)
        add_custom_command(TARGET mindspore-lite-jni POST_BUILD COMMAND ${NDK_STRIP}
                ${CMAKE_BINARY_DIR}/libmindspore-lite*.so)
    elseif(NOT WIN32)
        add_custom_command(TARGET mindspore-lite-jni POST_BUILD COMMAND strip
                ${CMAKE_BINARY_DIR}/libmindspore-lite*.so)
    endif()
endif()

