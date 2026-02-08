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
project(pass VERSION 1.0.0 LANGUAGES CXX)

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

# Set CMake module path - adjusted for mindspore test location
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Detect MindSpore HashMap type
include(find_ms_hash_map)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/replace_addn_fusion)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/add_neg_fusion)

# Handle multiple MindSpore include directories
if(MINDSPORE_INCLUDE_DIRS)
    # If contains multiple paths (semicolon separated), add them separately
    string(REPLACE ";" " " MINDSPORE_INCLUDE_DIRS_LIST "${MINDSPORE_INCLUDE_DIRS}")
    foreach(include_dir ${MINDSPORE_INCLUDE_DIRS_LIST})
        # Add complete MindSpore include paths to ensure all dependency headers are found
        include_directories(${include_dir})
        include_directories(${include_dir}/mindspore)
        include_directories(${include_dir}/mindspore/core/include)
        # Add MindSpore ccsrc path, contains backend/common/pass_manager/pass.h
        include_directories(${include_dir}/mindspore/ccsrc/)
        # Add MindSpore custom_pass path, contains custom_pass_plugin.h
        include_directories(${include_dir}/mindspore/ccsrc/include/backend/common/custom_pass/)
        # Add MindSpore ops path, contains common/device_type.h
        include_directories(${include_dir}/mindspore/ops)
        include_directories(${include_dir}/mindspore/ops/include)
        include_directories(${include_dir}/mindspore/ops/kernel/include)
        # Add third_party path, contains securec.h
        include_directories(${include_dir}/third_party)
        # Add specific securec path
        include_directories(${include_dir}/third_party/securec/include)
    endforeach()
endif()

# Link directories
link_directories(${MINDSPORE_LIB_DIRS})

# Automatically find all source files
file(GLOB_RECURSE PASS_SOURCES "*.cc")
file(GLOB_RECURSE PASS_HEADERS "*.h")

# Create dynamic library (based on installed MindSpore)
add_library(pass SHARED ${PASS_SOURCES})

# Link MindSpore libraries (based on actual requirements)
target_link_libraries(pass
    ${MINDSPORE_LIB_DIRS}/libmindspore_backend_common.so
    ${MINDSPORE_LIB_DIRS}/libmindspore_core.so
    ${MINDSPORE_LIB_DIRS}/libmindspore_common.so
)

# Default settings
option(ENABLE_GLIBCXX "enable_glibcxx" OFF)

# System-related overrides
if(NOT CMAKE_SYSTEM_NAME MATCHES "Linux")
    set(ENABLE_GLIBCXX ON)
endif()

# Environment variable overrides
if(DEFINED ENV{ENABLE_GLIBCXX})
    set(ENABLE_GLIBCXX $ENV{ENABLE_GLIBCXX})
endif()

# ABI flag settings
if(CMAKE_SYSTEM_NAME MATCHES "Linux")
    if(NOT ENABLE_GLIBCXX)
        add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)
    endif()
endif()

# Set compilation options
target_compile_options(pass PRIVATE
    -fPIC
    -std=c++17
    -Wall
    -Wextra
)

# Use ABI settings consistent with MindSpore
if(CMAKE_SYSTEM_NAME MATCHES "Linux")
    if(NOT ENABLE_GLIBCXX)
        target_compile_definitions(pass PRIVATE _GLIBCXX_USE_CXX11_ABI=0)
    endif()
endif()

# Set compilation definitions
target_compile_definitions(pass PRIVATE
    -DPASS_PLUGIN_EXPORTS
    -DMINDSPORE_PASS
)

# Set dynamic library properties
set_target_properties(pass PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    PREFIX "lib"
    OUTPUT_NAME "pass"
)

# Installation rules
install(TARGETS pass
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)
