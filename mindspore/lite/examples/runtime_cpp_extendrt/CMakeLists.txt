cmake_minimum_required(VERSION 3.14)
project(RuntimeCpp)
option(SUPPORT_NPU "if support npu" off)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 7.3.0)
    message(FATAL_ERROR "GCC version ${CMAKE_CXX_COMPILER_VERSION} must not be less than 7.3.0")
endif()

if(DEFINED ENV{LITE_HOME})
    set(LITE_HOME $ENV{LITE_HOME})
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Add directory to include search path
include_directories(${LITE_HOME}/runtime)
# Add directory to linker search path
link_directories(${LITE_HOME}/runtime/lib)
link_directories(${LITE_HOME}/tools/converter/lib)

link_directories(${CMAKE_CURRENT_SOURCE_DIR}/lib)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")

file(GLOB_RECURSE RUNTIME_CPP ${CMAKE_CURRENT_SOURCE_DIR}/*.cc)

add_executable(runtime_cpp ${RUNTIME_CPP})
target_link_libraries(runtime_cpp mindspore-lite pthread dl)
