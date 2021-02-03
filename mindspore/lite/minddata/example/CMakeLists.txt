cmake_minimum_required(VERSION 3.14.1)
project(testlenet)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I/usr/local/include -std=c++17 -Werror
-Wall -Wno-deprecated-declarations -fPIC")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-sign-compare")

set(DepDIR "${CMAKE_CURRENT_SOURCE_DIR}/mindspore-lite-1.1.0-inference-linux-x64/minddata")

include_directories(${DepDIR})



add_executable(testlenet
        ${CMAKE_CURRENT_SOURCE_DIR}/testlenet.cpp
        )

target_link_libraries(testlenet
        ${DepDIR}/lib/libminddata-lite.so
        ${DepDIR}/third_party/libjpeg-turbo/lib/libjpeg.so.62
        ${DepDIR}/third_party/libjpeg-turbo/lib/libturbojpeg.so.0
        pthread)