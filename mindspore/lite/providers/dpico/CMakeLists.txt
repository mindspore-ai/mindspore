cmake_minimum_required(VERSION 3.12)
project(DPICO_Custom)

if(DEFINED ENV{HISI_SDK_PATH})
    set(SD3403_SDK_PATH $ENV{HISI_SDK_PATH}/sd3403_sdk/dpico_acl_adapter/3403_sdk_0924)
else()
    message(FATAL_ERROR "error, no sdk found")
endif()

include_directories(BEFORE ${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${SD3403_SDK_PATH})
include_directories(${SD3403_SDK_PATH}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../runtime)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../runtime/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../runtime/include/third_party)
link_directories(${SD3403_SDK_PATH}/lib)

set(DEPENDENCY_LIBS
        ${SD3403_SDK_PATH}/lib/libsvp_acl.so
        ${SD3403_SDK_PATH}/lib/libprotobuf-c.so.1
        ${SD3403_SDK_PATH}/lib/libsecurec.so)

file(GLOB_RECURSE DPICO_RUNTIME_SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/common/*.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/infer/*.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/manager/*.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc
        )

add_library(dpico_acl_adapter SHARED ${DPICO_RUNTIME_SRC})
target_link_libraries(dpico_acl_adapter ${DEPENDENCY_LIBS})
