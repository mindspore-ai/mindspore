cmake_minimum_required(VERSION 3.14)
project(RuntimeGPUExtendTutorial)

message(STATUS "Using toolchain file: ${CMAKE_TOOLCHAIN_FILE}.")

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 7.3.0)
    message(FATAL_ERROR "GCC version ${CMAKE_CXX_COMPILER_VERSION} must not be less than 7.3.0")
endif()

add_definitions(-DCL_TARGET_OPENCL_VERSION=200)
add_definitions(-DCL_HPP_TARGET_OPENCL_VERSION=120)
add_definitions(-DCL_HPP_MINIMUM_OPENCL_VERSION=120)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")

include(${CMAKE_CURRENT_SOURCE_DIR}/../../../../cmake/utils.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/../../../../cmake/external_libs/opencl.cmake)
gene_opencl(${CMAKE_CURRENT_SOURCE_DIR}/src/cl)

# Add directory to include search path
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/runtime/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/runtime/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/runtime/include/third_party)
include_directories(${CMAKE_BINARY_DIR}/_deps/opencl-headers-src/)
include_directories(${CMAKE_BINARY_DIR}/_deps/opencl-clhpp-src/include)

# Add directory to linker search path
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/runtime/lib)

file(GLOB_RECURSE RUNTIME_REGISTRY_SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.cc)

add_executable(runtime_extend_tutorial ${RUNTIME_REGISTRY_SRC})
target_link_libraries(
        runtime_extend_tutorial
        mindspore-lite
        log
)

add_executable(runtime_extend_tutorial_static ${RUNTIME_REGISTRY_SRC})
target_link_libraries(
        runtime_extend_tutorial_static
        -Wl,--whole-archive libmindspore-lite.a -Wl,--no-whole-archive
        log
)
