/**
 * Copyright 2021 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "coder/generator/component/const_blocks/cmake_lists.h"

namespace mindspore::lite::micro {

const char *bench_cmake_lists_txt = R"RAW(
cmake_minimum_required(VERSION 3.14)
project(benchmark)

if(NOT DEFINED MODEL_LIB)
    message(FATAL_ERROR "MODEL_LIB not set")
endif()

get_filename_component(MODEL_LIB ${MODEL_LIB} ABSOLUTE BASE_DIR ${CMAKE_CURRENT_BINARY_DIR})

function(parse_lib_info lib_full_path lib_name lib_path)
    string(FIND "${lib_full_path}" "/" POS REVERSE)
    math(EXPR POS "${POS} + 1")
    string(SUBSTRING ${lib_full_path} 0 ${POS} path)
    set(${lib_path} ${path} PARENT_SCOPE)
    string(SUBSTRING ${lib_full_path} "${POS}" "-1" name)
    set(${lib_name} ${name} PARENT_SCOPE)
endfunction(parse_lib_info)

parse_lib_info(${MODEL_LIB} MODEL_LIB_NAME MODEL_LIB_PATH)

message("project name: ${MODEL_LIB_NAME}")

option(MICRO_BUILD_ARM64 "build android arm64" OFF)
option(MICRO_BUILD_ARM32A "build android arm32" OFF)

if(MICRO_BUILD_ARM64 OR MICRO_BUILD_ARM32A)
  add_compile_definitions(ENABLE_NEON)
  add_compile_definitions(ENABLE_ARM)
endif()

if(MICRO_BUILD_ARM64)
  add_compile_definitions(ENABLE_ARM64)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=armv8.2-a+dotprod")
endif()

if(MICRO_BUILD_ARM32A)
  add_compile_definitions(ENABLE_ARM32)
  add_definitions(-mfloat-abi=softfp -mfpu=neon)
endif()

set(CMAKE_C_FLAGS "${CMAKE_ENABLE_C99} ${CMAKE_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    message(STATUS "build benchmark with debug info")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DDebug -g")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DDebug -g")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fvisibility=default")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=default")
else()
    set(CMAKE_C_FLAGS "-fPIC -fPIE -D_FORTIFY_SOURCE=2 -O2 -Wall -Werror -fstack-protector-strong -Wno-attributes \
    -Wno-deprecated-declarations -Wno-missing-braces ${CMAKE_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "-fPIC -fPIE -D_FORTIFY_SOURCE=2 -O2 -Wall -Werror -fstack-protector-strong -Wno-attributes \
    -Wno-deprecated-declarations -Wno-missing-braces -Wno-overloaded-virtual ${CMAKE_CXX_FLAGS}")
endif()
link_directories(${MODEL_LIB_PATH})
include(benchmark.cmake)
add_executable(benchmark ${SRC_FILES})
target_link_libraries(benchmark ${MODEL_LIB_NAME} -lm -pthread)

)RAW";

const char *src_cmake_lists_txt = R"RAW(
cmake_minimum_required(VERSION 3.14)
project(net)

if(NOT DEFINED OP_LIB)
    message(FATAL_ERROR "OP_LIB not set")
endif()

if(NOT DEFINED OP_HEADER_PATH)
    message(FATAL_ERROR "OP_HEADER_PATH not set")
endif()

get_filename_component(OP_LIB ${OP_LIB} ABSOLUTE BASE_DIR ${CMAKE_CURRENT_BINARY_DIR})
get_filename_component(OP_HEADER_PATH ${OP_HEADER_PATH} ABSOLUTE BASE_DIR ${CMAKE_CURRENT_BINARY_DIR})

message("operator lib path: ${OP_LIB}")
message("operator header path: ${OP_HEADER_PATH}")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../include)
include_directories(${OP_HEADER_PATH})
include_directories(${HEADER_PATH})

include(net.cmake)

option(MICRO_BUILD_ARM64 "build android arm64" OFF)
option(MICRO_BUILD_ARM32A "build android arm32" OFF)

if(MICRO_BUILD_ARM64 OR MICRO_BUILD_ARM32A)
  add_compile_definitions(ENABLE_NEON)
  add_compile_definitions(ENABLE_ARM)
endif()

if(MICRO_BUILD_ARM64)
  add_compile_definitions(ENABLE_ARM64)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=armv8.2-a+dotprod")
endif()

if(MICRO_BUILD_ARM32A)
  add_compile_definitions(ENABLE_ARM32)
  add_definitions(-mfloat-abi=softfp -mfpu=neon)
endif()

set(CMAKE_C_FLAGS "${CMAKE_ENABLE_C99} ${CMAKE_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DDebug -g")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DDebug -g")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fvisibility=default")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=default")
else()
    set(CMAKE_C_FLAGS "-fPIC -fPIE -D_FORTIFY_SOURCE=2 -O2 -Wall -Werror -fstack-protector-strong -Wno-attributes \
    -Wno-deprecated-declarations -Wno-missing-braces ${CMAKE_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "-fPIC -fPIE -D_FORTIFY_SOURCE=2 -O2 -Wall -Werror -fstack-protector-strong -Wno-attributes \
    -Wno-deprecated-declarations -Wno-missing-braces -Wno-overloaded-virtual ${CMAKE_CXX_FLAGS}")
endif()

function(create_library)
    add_custom_command(TARGET net
            POST_BUILD
            COMMAND rm -rf tmp
            COMMAND mkdir tmp
            COMMAND cd tmp && ar -x ${OP_LIB}
            COMMAND echo "raw static library ${library_name} size:"
            COMMAND ls -lh ${library_name}
            COMMAND mv ${library_name} ./tmp && cd tmp && ar -x ${library_name}
            COMMENT "unzip raw static library ${library_name}"
            )
    foreach(object_file ${OP_SRC})
        add_custom_command(TARGET net POST_BUILD COMMAND mv ./tmp/${object_file} .)
    endforeach()
    add_custom_command(TARGET net
            POST_BUILD
            COMMAND ar cr ${library_name} *.o
            COMMAND ranlib ${library_name}
            COMMAND echo "new static library ${library_name} size:"
            COMMAND ls -lh ${library_name}
            COMMAND rm -rf tmp && rm -rf *.o
            COMMENT "generate specified static library ${library_name}"
            )
endfunction(create_library)
string(CONCAT library_name "lib" net ".a")
create_library()

)RAW";

}  // namespace mindspore::lite::micro
