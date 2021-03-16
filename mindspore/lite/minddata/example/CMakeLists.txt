cmake_minimum_required(VERSION 3.14.1)
project(testlenet)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror -Wall -fPIC -std=c++17")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-sign-compare")

set(MD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mindspore-lite-1.2.0-inference-linux-x64/minddata")
set(MS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mindspore-lite-1.2.0-inference-linux-x64/")

include_directories(${MD_DIR})
include_directories(${MS_DIR})


add_executable(testlenet
        ${CMAKE_CURRENT_SOURCE_DIR}/testlenet.cpp
        )

target_link_libraries(testlenet
        ${MD_DIR}/lib/libminddata-lite.so
        ${MD_DIR}/third_party/libjpeg-turbo/lib/libjpeg.so.62
        ${MD_DIR}/third_party/libjpeg-turbo/lib/libturbojpeg.so.0
        ${MS_DIR}/lib/libmindspore-lite.so
        pthread)

add_executable(testresize
        ${CMAKE_CURRENT_SOURCE_DIR}/testresize.cpp
        )

target_link_libraries(testresize
        ${MD_DIR}/lib/libminddata-lite.so
        ${MD_DIR}/third_party/libjpeg-turbo/lib/libjpeg.so.62
        ${MD_DIR}/third_party/libjpeg-turbo/lib/libturbojpeg.so.0
        ${MS_DIR}/lib/libmindspore-lite.so
        pthread)