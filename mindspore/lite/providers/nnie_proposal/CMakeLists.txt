cmake_minimum_required(VERSION 3.12)
project(NNIE_proposal)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../runtime)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../runtime/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../runtime/include/third_party)

aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/src COMMON_SRC3)

add_library(mslite_proposal SHARED ${COMMON_SRC3})
target_link_libraries(mslite_proposal ${LINK_LOCAT_LIB})
