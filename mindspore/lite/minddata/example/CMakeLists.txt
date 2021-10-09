cmake_minimum_required(VERSION 3.14.1)
project(testlenet)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror -Wall -fPIC -std=c++17")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-sign-compare")

set(MS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mindspore-lite-1.5.0-linux-x64/runtime")
set(LITECV_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mindspore-lite-1.5.0-linux-x64/runtime/include/dataset")

include_directories(${MS_DIR} ${LITECV_DIR})


add_executable(testlenet
        ${CMAKE_CURRENT_SOURCE_DIR}/testlenet.cpp
        )

target_link_libraries(testlenet
        ${MS_DIR}/lib/libminddata-lite.so
        ${MS_DIR}/third_party/libjpeg-turbo/lib/libjpeg.so.62
        ${MS_DIR}/third_party/libjpeg-turbo/lib/libturbojpeg.so.0
        ${MS_DIR}/lib/libmindspore-lite.so
        pthread)

add_executable(testresize
        ${CMAKE_CURRENT_SOURCE_DIR}/testresize.cpp
        )

target_link_libraries(testresize
        ${MS_DIR}/lib/libminddata-lite.so
        ${MS_DIR}/third_party/libjpeg-turbo/lib/libjpeg.so.62
        ${MS_DIR}/third_party/libjpeg-turbo/lib/libturbojpeg.so.0
        ${MS_DIR}/lib/libmindspore-lite.so
        pthread)

add_executable(testlitecv
        ${CMAKE_CURRENT_SOURCE_DIR}/testlitecv.cpp
        )

target_link_libraries(testlitecv
        ${MS_DIR}/lib/libminddata-lite.so
        ${MS_DIR}/third_party/libjpeg-turbo/lib/libjpeg.so.62
        ${MS_DIR}/third_party/libjpeg-turbo/lib/libturbojpeg.so.0
        pthread)