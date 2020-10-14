cmake_minimum_required(VERSION 3.14)
project (Lite-java)

set(MS_VERSION_MAJOR ${MS_VERSION_MAJOR})
set(MS_VERSION_MINOR ${MS_VERSION_MINOR})
set(MS_VERSION_REVISION ${MS_VERSION_REVISION})
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DMS_VERSION_MAJOR=${MS_VERSION_MAJOR} -DMS_VERSION_MINOR=${MS_VERSION_MINOR} -DMS_VERSION_REVISION=${MS_VERSION_REVISION}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DMS_VERSION_MAJOR=${MS_VERSION_MAJOR} -DMS_VERSION_MINOR=${MS_VERSION_MINOR} -DMS_VERSION_REVISION=${MS_VERSION_REVISION}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")

if (ENABLE_VERBOSE)
    set(CMAKE_VERBOSE_MAKEFILE on)
endif ()

if (PLATFORM_ARM32)
    add_compile_definitions(ENABLE_ARM32)
endif ()
if (PLATFORM_ARM64)
    add_compile_definitions(ENABLE_ARM64)
endif ()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../../../../../)  ## lite include
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../../../../../../core/) ## core include
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../../../../../../../third_party/flatbuffers/include) ## third_party
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../../libs/${ANDROID_ABI}/)

add_library(mindspore-lite-jni SHARED
        ${CMAKE_CURRENT_SOURCE_DIR}/common/jni_utils.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/runtime/model.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/runtime/version.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/runtime/ms_config.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/runtime/ms_tensor.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/runtime/lite_session.cpp
        )

find_library(log-lib log)

target_link_libraries(mindspore-lite-jni mindspore-lite ${log-lib} )