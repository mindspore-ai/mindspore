cmake_minimum_required(VERSION 3.14)
project(QuickStartCpp)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 7.3.0)
    message(FATAL_ERROR "GCC version ${CMAKE_CXX_COMPILER_VERSION} must not be less than 7.3.0")
endif()

if(DEFINED ENV{LITE_HOME})
    set(LITE_HOME $ENV{LITE_HOME})
endif()

include_directories(${LITE_HOME}/runtime)
link_directories(${LITE_HOME}/runtime/lib)
link_directories(${LITE_HOME}/tools/converter/lib)

file(GLOB_RECURSE GPU_TRT_DISTRIBUTED ${CMAKE_CURRENT_SOURCE_DIR}/*.cc)
add_executable(gpu_trt_distributed ${GPU_TRT_DISTRIBUTED})

target_link_libraries(gpu_trt_distributed mindspore-lite)
