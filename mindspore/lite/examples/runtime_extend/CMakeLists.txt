cmake_minimum_required(VERSION 3.14)
project(RuntimeExtendTutorial)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 7.3.0)
    message(FATAL_ERROR "GCC version ${CMAKE_CXX_COMPILER_VERSION} must not be less than 7.3.0")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")

# Add directory to include search path
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/third_party)

# Add directory to linker search path
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/lib)

file(GLOB_RECURSE RUNTIME_REGISTRY_SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.cc)
add_executable(runtime_extend_tutorial ${RUNTIME_REGISTRY_SRC})

target_link_libraries(
        runtime_extend_tutorial
        mindspore-lite
        pthread
        dl
)
