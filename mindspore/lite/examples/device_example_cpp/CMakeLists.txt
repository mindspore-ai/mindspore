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
    mindspore_quick_start_cpp
    -Wl,--whole-archive mindspore-lite -Wl,--no-whole-archive
    pthread
)

if("${EXAMPLE_TARGET}" STREQUAL "Ascend")
    target_link_libraries(mindspore_quick_start_cpp ascendcl)
else()
    target_link_libraries(mindspore_quick_start_cpp cudart cublas)
endif()

# Due to the increased compilation options for stack protection,
# it is necessary to target link ssp library when Use the static library in Windows.
if(WIN32)
    target_link_libraries(
        mindspore_quick_start_cpp
        ssp
    )
else()
    target_link_libraries(
        mindspore_quick_start_cpp
        dl
    )
endif()

