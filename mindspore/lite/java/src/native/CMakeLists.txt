cmake_minimum_required(VERSION 3.4.1)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/libs/${ANDROID_ABI})

add_library(mindspore-lite-jni SHARED
        ${CMAKE_CURRENT_SOURCE_DIR}/src/common/ms_log.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/common/jni_utils.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/runtime/model.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/runtime/context.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/runtime/ms_tensor.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/runtime/lite_session.cpp
        )

find_library(log-lib log)

target_link_libraries(mindspore-lite-jni mindspore-lite ${log-lib} )