cmake_minimum_required(VERSION 3.14)
project(DPICO_Custom)
include(${CMAKE_CURRENT_SOURCE_DIR}/../../../../../cmake/utils.cmake)

if(DEFINED ENV{HISI_SDK_PATH})
    set(SD3403_SDK_PATH $ENV{HISI_SDK_PATH}/sd3403_sdk/dpico_acl_adapter/3403_sdk)
else()
    message(FATAL_ERROR "error, no sdk found")
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${SD3403_SDK_PATH})
include_directories(${SD3403_SDK_PATH}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party/runtime)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party/runtime/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party/runtime/include/third_party)
link_directories(${SD3403_SDK_PATH}/lib)

aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/src COMMON_SRC)

set(LINK_LOCAT_LIB ${SD3403_SDK_PATH}/lib/libsvp_acl.so
        pthread ${SD3403_SDK_PATH}/lib/libsecurec.so dl
        ${SD3403_SDK_PATH}/lib/libprotobuf-c.so.1 stdc++)

add_library(dpico_acl_adapter SHARED
        ${COMMON_SRC})
target_link_libraries(dpico_acl_adapter ${LINK_LOCAT_LIB} securec)