# Copyright 2025 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

cmake_minimum_required(VERSION 3.16)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find MindSpore (supports both installed and source versions)
if(DEFINED MINDSPORE_ROOT)
    # Use specified MindSpore path
    set(MINDSPORE_INCLUDE_DIRS ${MINDSPORE_ROOT}/include)
    set(MINDSPORE_LIB_DIRS ${MINDSPORE_ROOT}/lib)
    message(STATUS "Using MindSpore from: ${MINDSPORE_ROOT}")
else()
    # Try to find installed MindSpore
    find_package(MindSpore QUIET)
    if(NOT MindSpore_FOUND)
        message(WARNING "MindSpore not found, using default paths")
        set(MINDSPORE_INCLUDE_DIRS "/usr/local/include")
        set(MINDSPORE_LIB_DIRS "/usr/local/lib")
    endif()
endif()

# Build options configuration (simplified)
set(CMAKE_BUILD_TYPE "Release")
set(BUILD_SHARED_LIBS ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Add detailed compilation logs
set(CMAKE_VERBOSE_MAKEFILE ON)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Handle multiple MindSpore include directories
if(MINDSPORE_INCLUDE_DIRS)
    # If contains multiple paths (semicolon separated), add them separately
    string(REPLACE ";" " " MINDSPORE_INCLUDE_DIRS_LIST "${MINDSPORE_INCLUDE_DIRS}")
    foreach(include_dir ${MINDSPORE_INCLUDE_DIRS_LIST})
        include_directories(${include_dir})
        # Add complete MindSpore include paths to ensure all dependency headers are found
        include_directories(${include_dir}/mindspore)
        include_directories(${include_dir}/mindspore/core/include)
        # Add MindSpore ccsrc path, contains mindspore/ccsrc/include/
        include_directories(${include_dir}/mindspore/ccsrc/include)
        # Add MindSpore csrc path, contains mindspore/ccsrc/
        include_directories(${include_dir}/mindspore/ccsrc)
        # Add third_party path, contains securec.h
        include_directories(${include_dir}/third_party)
        include_directories(${include_dir}/third_party/include)
        # Add specific pybind11 path
        include_directories(${include_dir}/third_party/pybind11)
    endforeach()
endif()

# Find Python
find_package(Python3 COMPONENTS Interpreter Development)
if(Python3_FOUND)
    set(PYTHON_INCLUDE_DIRS "${Python3_INCLUDE_DIRS}")
    set(PYTHON_LIBRARIES "${Python3_LIBRARIES}")
    if(WIN32)
        if(Python3_DIR)
            message("Python3_DIR set already: " ${Python3_DIR})
        else()
            string(LENGTH ${PYTHON_LIBRARIES} PYTHON_LIBRARIES_LEN)
            string(LENGTH "libpythonxx.a" Python3_NAME_LEN)
            math(EXPR Python3_DIR_LEN  ${PYTHON_LIBRARIES_LEN}-${Python3_NAME_LEN})
            string(SUBSTRING ${Python3_LIBRARIES} 0 ${Python3_DIR_LEN} Python3_DIR)
            message("Python3_DIR: " ${Python3_DIR})
        endif()
        link_directories(${Python3_DIR})
    endif()
else()
    find_python_package(py_inc py_lib)
    set(PYTHON_INCLUDE_DIRS "${py_inc}")
    set(PYTHON_LIBRARIES "${py_lib}")
endif()
include_directories(${PYTHON_INCLUDE_DIRS})

# Link directories
link_directories(${MINDSPORE_LIB_DIRS})

# Automatically find all source files
file(GLOB_RECURSE SRC_SOURCES "*.cc")

# Create dynamic library (based on installed MindSpore)
add_library(custom_backend SHARED ${SRC_SOURCES})

# Link MindSpore libraries (based on actual requirements)
target_link_libraries(custom_backend
    ${MINDSPORE_LIB_DIRS}/libmindspore_backend_manager.so
    ${MINDSPORE_LIB_DIRS}/libmindspore_core.so
    ${MINDSPORE_LIB_DIRS}/libmindspore_common.so
)

# ABI flag settings
if(CMAKE_SYSTEM_NAME MATCHES "Linux")
    if(NOT ENABLE_GLIBCXX)
        add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)
    endif()
endif()

# Set compilation options
target_compile_options(custom_backend PRIVATE
    -fPIC
    -std=c++17
    -Wall
    -Wextra
)

# Installation rules
install(TARGETS custom_backend
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)
