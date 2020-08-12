cmake_minimum_required(VERSION 3.14)
project (Lite-java)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../../lite/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../../core/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../../../third_party/flatbuffers/include)
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/../lib)

add_library(mindspore-lite-jni SHARED
        ${CMAKE_CURRENT_SOURCE_DIR}/common/ms_log.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/common/jni_utils.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/runtime/model.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/runtime/context.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/runtime/ms_tensor.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/runtime/lite_session.cpp
        )

find_library(log-lib log)

target_link_libraries(mindspore-lite-jni mindspore-lite ${log-lib} )