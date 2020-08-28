cmake_minimum_required(VERSION 3.14)
project (Lite-java)

set(MS_VERSION_MAJOR 0)
set(MS_VERSION_MINOR 7)
set(MS_VERSION_REVISION 0)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DMS_VERSION_MAJOR=${MS_VERSION_MAJOR} -DMS_VERSION_MINOR=${MS_VERSION_MINOR} -DMS_VERSION_REVISION=${MS_VERSION_REVISION}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DMS_VERSION_MAJOR=${MS_VERSION_MAJOR} -DMS_VERSION_MINOR=${MS_VERSION_MINOR} -DMS_VERSION_REVISION=${MS_VERSION_REVISION}")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../../lite/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../../core/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../../../third_party/flatbuffers/include)
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/../lib)

add_library(mindspore-lite-jni SHARED
        ${CMAKE_CURRENT_SOURCE_DIR}/common/jni_utils.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/runtime/model.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/runtime/version.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/runtime/context.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/runtime/ms_tensor.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/runtime/lite_session.cpp
        )

find_library(log-lib log)

target_link_libraries(mindspore-lite-jni mindspore-lite ${log-lib} )