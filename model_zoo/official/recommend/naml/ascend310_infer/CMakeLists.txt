# Copyright (c) Huawei Technologies Co., Ltd. 2019. All rights reserved.

# CMake lowest version requirement
cmake_minimum_required(VERSION 3.5.1)

# project information
project(ACL_RESNET50)

find_package(gflags REQUIRED)
include_directories(${gflags_INCLUDE_DIR})

# Compile options
add_compile_options(-std=c++11 -g -O0)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY  "../out")
set(CMAKE_CXX_FLAGS_DEBUG "-fPIC -O0 -g -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "-fPIC -O0 -g -Wall")

set(INC_PATH $ENV{DDK_PATH})

if(NOT DEFINED ENV{DDK_PATH})
    set(INC_PATH "/usr/local/Ascend")
    message(STATUS "set default INC_PATH: ${INC_PATH}")
else()
    message(STATUS "env INC_PATH: ${INC_PATH}")
endif()

set(LIB_PATH $ENV{NPU_HOST_LIB})

if(NOT DEFINED ENV{NPU_HOST_LIB})
    set(LIB_PATH "/usr/local/Ascend/acllib/lib64/stub/")
    message(STATUS "set default LIB_PATH: ${LIB_PATH}")
else()
    message(STATUS "env LIB_PATH: ${LIB_PATH}")
endif()

# Header path
include_directories(
    ${INC_PATH}/acllib/include/
    ../include/
)

if(target STREQUAL "Simulator_Function")
    add_compile_options(-DFUNC_SIM)
endif()

# add host lib path
link_directories(
    ${LIB_PATH}
)

# Set output directory
set(PROJECT_SRC_ROOT ${CMAKE_CURRENT_LIST_DIR}/)

# Set include directory and library directory
set(ACL_LIB_DIR $ENV{ASCEND_HOME}/acllib)
set(ATLAS_ACL_LIB_DIR $ENV{ASCEND_HOME}/ascend-toolkit/latest/acllib)
# Header path
include_directories(${ACL_LIB_DIR}/include/)
include_directories(${ATLAS_ACL_LIB_DIR}/include/)
include_directories(${PROJECT_SRC_ROOT}/../inc)

# add host lib path
link_directories(${ACL_LIB_DIR})
find_library(acl libascendcl.so ${ACL_LIB_DIR}/lib64 ${ATLAS_ACL_LIB_DIR}/lib64)

add_executable(main
        ./src/utils.cpp
        ./src/model_process.cpp
        ./src/sample_process.cpp
        ./src/main.cpp)

target_link_libraries(main ${acl} stdc++ gflags)

install(TARGETS main DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
