# Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.

# CMake lowest version requirement
cmake_minimum_required(VERSION 3.5.1)
# project information
project(InferClassification)
# Check environment variable
if(NOT DEFINED ENV{ASCEND_HOME})
    message(FATAL_ERROR "please define environment variable:ASCEND_HOME")
endif()

# Compile options
add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g -std=c++17 -Werror -Wall -fPIE -Wl,--allow-shlib-undefined")

# Skip build rpath
set(CMAKE_SKIP_BUILD_RPATH True)

# Set output directory
set(PROJECT_SRC_ROOT ${CMAKE_CURRENT_LIST_DIR}/)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SRC_ROOT}/../out)

# Set include directory and library directory
set(FWKACL_LIB_DIR $ENV{ASCEND_HOME}/fwkacllib)
set(ACL_LIB_DIR $ENV{ASCEND_HOME}/acllib)
set(ATLAS_ACL_LIB_DIR $ENV{ASCEND_HOME}/ascend-toolkit/latest/acllib)

# Header path
include_directories(${ACL_LIB_DIR}/include/)
include_directories(${FWKACL_LIB_DIR}/include/)
include_directories(${ATLAS_ACL_LIB_DIR}/include/)
include_directories(${PROJECT_SRC_ROOT}/../inc)

# add host lib path
link_directories(${ACL_LIB_DIR} ${FWKACL_LIB_DIR})
find_library(acl libascendcl.so ${ACL_LIB_DIR}/lib64 ${FWKACL_LIB_DIR}/lib64 ${ATLAS_ACL_LIB_DIR}/lib64)

add_executable(main utils.cpp
        sample_process.cpp
        model_process.cpp
        main.cpp)

target_link_libraries(main ${acl} gflags pthread)
