cmake_minimum_required(VERSION 3.14)
project(QuickStartCpp)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 7.3.0)
    message(FATAL_ERROR "GCC version ${CMAKE_CXX_COMPILER_VERSION} must not be less than 7.3.0")
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

if(DEFINED ENV{LITE_HOME})
    set(LITE_HOME $ENV{LITE_HOME})
endif()

if(DEFINED ENV{EXAMPLE_TARGET})
    set(EXAMPLE_TARGET $ENV{EXAMPLE_TARGET})
endif()

# Add directory to include search path
include_directories(${LITE_HOME}/runtime)
# Add directory to linker search path
link_directories(${LITE_HOME}/runtime/lib)

if("${EXAMPLE_TARGET}" STREQUAL "Ascend")
    include_directories(/usr/local/Ascend/latest/fwkacllib/include)
    link_directories(/usr/local/Ascend/latest/fwkacllib/lib64)
    add_definitions(-DENABLE_ASCEND)
else()
    set(CUDA_HOME $ENV{CUDA_HOME})
    include_directories(${CUDA_HOME}/include)
    link_directories(${CUDA_HOME}/lib64)
    add_definitions(-DENABLE_GPU)
endif()

file(GLOB_RECURSE QUICK_START_CXX ${CMAKE_CURRENT_SOURCE_DIR}/*.cc)
add_executable(mindspore_quick_start_cpp ${QUICK_START_CXX})

target_link_libraries(
    mindspore_quick_start_cpp mindspore-lite pthread dl
)

if("${EXAMPLE_TARGET}" STREQUAL "Ascend")
    target_link_libraries(mindspore_quick_start_cpp ascendcl)
else()
    target_link_libraries(mindspore_quick_start_cpp cudart cublas)
endif()
