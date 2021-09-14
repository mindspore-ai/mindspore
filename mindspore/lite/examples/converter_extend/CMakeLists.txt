cmake_minimum_required(VERSION 3.14)
project(ConverterExtendTutorial)
add_definitions(-DUSE_GLOG)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 7.3.0)
    message(FATAL_ERROR "GCC version ${CMAKE_CXX_COMPILER_VERSION} must not be less than 7.3.0")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")

# Add directory to include search path
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/core)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/third_party)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/third_party/eigen3)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/third_party/securec)

file(GLOB_RECURSE CONVERTER_REGISTRY_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc)
add_library(converter_extend_tutorial SHARED ${CONVERTER_REGISTRY_SRC})
