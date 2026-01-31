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

# FindMindSporeHashMap.cmake
# Detect MindSpore HashMap type and maintain consistency

message(STATUS "=== Starting MindSpore HashMap type detection ===")

# Method 1: Find MindSpore installation directory and library files
execute_process(
    COMMAND python3 -c "import mindspore; print(mindspore.__file__)"
    OUTPUT_VARIABLE MINDSPORE_MODULE_PATH
    ERROR_VARIABLE MINDSPORE_MODULE_ERROR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

message(STATUS "Python module path: ${MINDSPORE_MODULE_PATH}")
message(STATUS "Python error info: ${MINDSPORE_MODULE_ERROR}")

if(MINDSPORE_MODULE_PATH)
    # Derive library path from module path
    string(REPLACE "/__init__.py" "" MINDSPORE_ROOT "${MINDSPORE_MODULE_PATH}")
    string(REPLACE "/mindspore/__init__.py" "" MINDSPORE_ROOT "${MINDSPORE_ROOT}")

    message(STATUS "MindSpore root directory: ${MINDSPORE_ROOT}")

    # Find MindSpore library files - corrected path
    file(GLOB MINDSPORE_LIBS "${MINDSPORE_ROOT}/lib/libmindspore_*.so")

    message(STATUS "Found library files count: ${MINDSPORE_LIBS}")

    if(MINDSPORE_LIBS)
        # Select main library file for checking (usually libmindspore_core.so contains core functionality)
        list(FIND MINDSPORE_LIBS "${MINDSPORE_ROOT}/lib/libmindspore_core.so" CORE_LIB_INDEX)
        message(STATUS "libmindspore_core.so index: ${CORE_LIB_INDEX}")

        if(CORE_LIB_INDEX GREATER_EQUAL 0)
            list(GET MINDSPORE_LIBS ${CORE_LIB_INDEX} MINDSPORE_LIB)
        else()
            list(GET MINDSPORE_LIBS 0 MINDSPORE_LIB)
        endif()

        message(STATUS "Selected library file: ${MINDSPORE_LIB}")

        # Check if MindSpore library contains robin_hood symbols
        message(STATUS "Starting robin_hood symbol check...")

        # Method 1: Use bash to execute command
        execute_process(
            COMMAND bash -c "strings '${MINDSPORE_LIB}' | grep -i robin_hood | head -1"
            OUTPUT_VARIABLE ROBIN_HOOD_CHECK
            ERROR_VARIABLE ROBIN_HOOD_CHECK_ERROR
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )

        message(STATUS "robin_hood check result: '${ROBIN_HOOD_CHECK}'")
        message(STATUS "robin_hood check error: '${ROBIN_HOOD_CHECK_ERROR}'")

        # If bash method fails, try reading file directly
        if(NOT ROBIN_HOOD_CHECK)
            message(STATUS "bash method failed, trying to read file directly...")
            file(READ ${MINDSPORE_LIB} LIB_CONTENT)
            string(FIND "${LIB_CONTENT}" "robin_hood" ROBIN_HOOD_POS)
            if(ROBIN_HOOD_POS GREATER_EQUAL 0)
                set(ROBIN_HOOD_CHECK "found_in_file")
                message(STATUS "Found robin_hood in file content")
            endif()
        endif()

        if(ROBIN_HOOD_CHECK)
            message(STATUS "MindSpore uses robin_hood::unordered_map")

            # Check if robin_hood.h exists
            if(EXISTS "${MINDSPORE_ROOT}/include/third_party/robin_hood_hashing/include/robin_hood.h")
                message(STATUS "Found robin_hood.h: "
                        "${MINDSPORE_ROOT}/include/third_party/robin_hood_hashing/include/robin_hood.h")
                add_compile_definitions(ENABLE_FAST_HASH_TABLE=1)
                add_compile_definitions(HASHMAP_TYPE="robin_hood")
                # Add robin_hood header file path
                include_directories("${MINDSPORE_ROOT}/include/third_party/robin_hood_hashing")
                message(STATUS "Using fast hash table (robin_hood) for ms_custom_ops to match MindSpore")
            else()
                message(WARNING "robin_hood.h not found, falling back to std::unordered_map")
                add_compile_definitions(HASHMAP_TYPE="std")
                message(STATUS "Using standard hash table (std::unordered_map) for ms_custom_ops "
                        "(robin_hood.h not found)")
            endif()
        else()
            message(STATUS "MindSpore uses std::unordered_map")
            add_compile_definitions(HASHMAP_TYPE="std")
            message(STATUS "Using standard hash table (std::unordered_map) for ms_custom_ops to match MindSpore")
        endif()
    else()
        message(WARNING "MindSpore library not found in ${MINDSPORE_ROOT}/mindspore/lib/")
        set(MINDSPORE_LIB "")
    endif()
else()
    message(FATAL_ERROR "Could not find MindSpore Python module")
endif()

message(STATUS "=== MindSpore HashMap type detection completed ===")

# Add compile-time log definition (for conditional compilation)
add_compile_definitions(LOG_HASHMAP_TYPE)
