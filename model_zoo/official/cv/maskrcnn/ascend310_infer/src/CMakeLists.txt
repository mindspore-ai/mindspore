# Copyright (c) Huawei Technologies Co., Ltd. 2020. All rights reserved.

# CMake lowest version requirement
cmake_minimum_required(VERSION 3.5.1)
# Add definitions ENABLE_DVPP_INTERFACE to use dvpp api
add_definitions(-DENABLE_DVPP_INTERFACE)
# project information
project(InferClassification)
# Check environment variable
if(NOT DEFINED ENV{ASCEND_HOME})
    message(FATAL_ERROR "please define environment variable:ASCEND_HOME")
endif()

# Compile options
add_compile_options(-std=c++11 -fPIE -g -fstack-protector-all -Werror -Wreturn-type)

# Skip build rpath
set(CMAKE_SKIP_BUILD_RPATH True)

# Set output directory
set(PROJECT_SRC_ROOT ${CMAKE_CURRENT_LIST_DIR}/)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SRC_ROOT}/out)

# Set include directory and library directory
#set(ACL_INC_DIR $ENV{ASCEND_HOME}/$ENV{ASCEND_VERSION}/$ENV{ARCH_PATTERN}/include)
#set(ACL_LIB_DIR $ENV{ASCEND_HOME}/$ENV{ASCEND_VERSION}/$ENV{ARCH_PATTERN}/lib64/stub)
set(ACL_INC_DIR $ENV{ASCEND_HOME}/acllib/include)
set(ACL_LIB_DIR $ENV{ASCEND_HOME}/acllib/lib64/stub)
# Header path
include_directories(${ACL_INC_DIR})
include_directories(${PROJECT_SRC_ROOT}/../inc)

# add host lib path
link_directories(${ACL_LIB_DIR})

add_executable(main AclProcess.cpp
                    DvppCommon.cpp
                    ModelProcess.cpp
                    main.cpp)

target_link_libraries(main ascendcl gflags acl_dvpp pthread)
