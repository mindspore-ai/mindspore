cmake_minimum_required(VERSION 3.12)
project(NNIE_proposal)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../runtime)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../runtime/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../runtime/include/third_party)

aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/src COMMON_SRC3)

add_library(mslite_proposal SHARED ${COMMON_SRC3})
target_link_libraries(mslite_proposal ${LINK_LOCAT_LIB})

if(DEFINED HIMIX_STRIP)
    set(NDK_STRIP ${HIMIX_STRIP})
else()
    set(NDK_STRIP "arm-himix200-linux-strip")
endif()

if("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
    add_custom_command(TARGET mslite_proposal POST_BUILD COMMAND ${NDK_STRIP}
            ${CMAKE_CURRENT_BINARY_DIR}/libmslite_proposal.so)
endif()