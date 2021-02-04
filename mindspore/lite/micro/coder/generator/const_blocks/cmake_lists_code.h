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

#ifndef MICRO_LITE_MICRO_CODER_GENERATOR_CONST_BLOCKS_CMAKE_LISTS_CODE_H_
#define MICRO_LITE_MICRO_CODER_GENERATOR_CONST_BLOCKS_CMAKE_LISTS_CODE_H_

static const char bench_cmake_lists_txt[] =
  "cmake_minimum_required(VERSION 3.14)\n"
  "project(${PROJ_NAME})\n"
  "\n"
  "message(\"project name: ${PROJ_NAME}\")\n"
  "message(\"project name: ${MODEL_LIB_PATH}\")\n"
  "message(\"architecture cmake file path: ${ARCH_CMAKE_PATH}\")\n"
  "\n"
  "function(parse_lib_info lib_full_path lib_name lib_path)\n"
  "    string(FIND \"${lib_full_path}\" \"/\" POS REVERSE)\n"
  "    math(EXPR POS \"${POS} + 1\")\n"
  "    string(SUBSTRING ${lib_full_path} 0 ${POS} path)\n"
  "    set(${lib_path} ${path} PARENT_SCOPE)\n"
  "    string(SUBSTRING ${lib_full_path} \"${POS}\" \"-1\" name)\n"
  "    set(${lib_name} ${name} PARENT_SCOPE)\n"
  "endfunction(parse_lib_info)\n"
  "\n"
  "parse_lib_info(${MODEL_LIB} MODEL_LIB_NAME MODEL_LIB_PATH)\n"
  "\n"
  "if (\"${ARCH_CMAKE_PATH}\" STREQUAL \"\")\n"
  "    message(\"arch is x86_64\")\n"
  "else ()\n"
  "    include(${ARCH_CMAKE_PATH})\n"
  "endif ()\n"
  "\n"
  "include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../include)\n"
  "\n"
  "set(CMAKE_C_FLAGS \"${CMAKE_ENABLE_C99} ${CMAKE_C_FLAGS}\")\n"
  "if (\"${CMAKE_BUILD_TYPE}\" STREQUAL \"Debug\")\n"
  "    message(\"*******************${CMAKE_BUILD_TYPE}**********\")\n"
  "    set(CMAKE_C_FLAGS \"-DDebug -g -fPIC -fPIE -fvisibility=default ${CMAKE_C_FLAGS}\")\n"
  "else ()\n"
  "    set(CMAKE_C_FLAGS \"-fPIC -fPIE -O3 -fstack-protector-strong -fomit-frame-pointer ${CMAKE_C_FLAGS}\")\n"
  "    set(CMAKE_C_FLAGS_Release \"${CMAKE_C_FLAGS_Release} -O3 -ffunction-sections -fdata-sections\")\n"
  "endif ()\n"
  "link_directories(${MODEL_LIB_PATH})\n"
  "include(benchmark.cmake)\n"
  "add_executable(${PROJ_NAME}_bench ${SRC_FILES})\n"
  "target_link_libraries(${PROJ_NAME}_bench ${MODEL_LIB_NAME} -lm)\n";

static const char src_cmake_lists_txt[] =
  "cmake_minimum_required(VERSION 3.14)\n"
  "project(${PROJ_NAME})\n"
  "\n"
  "message(\"project name: ${PROJ_NAME}\")\n"
  "message(\"architecture cmake file path: ${ARCH_CMAKE_PATH}\")\n"
  "message(\"operator lib path: ${OP_LIB}\")\n"
  "message(\"operator header path: ${OP_HEADER_PATH}\")\n"
  "\n"
  "include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../include)\n"
  "include_directories(${OP_HEADER_PATH})\n"
  "\n"
  "include(net.cmake)\n"
  "\n"
  "if(\"${ARCH_CMAKE_PATH}\" STREQUAL \"\")\n"
  "    message(\"arch is x86_64\")\n"
  "else()\n"
  "    include(${ARCH_CMAKE_PATH})\n"
  "endif()\n"
  "\n"
  "set(CMAKE_C_FLAGS \"${CMAKE_ENABLE_C99} ${CMAKE_C_FLAGS}\")\n"
  "if(\"${CMAKE_BUILD_TYPE}\" STREQUAL \"Debug\")\n"
  "    set(CMAKE_C_FLAGS \"-DDebug -g -fPIC -fPIE -fvisibility=default ${CMAKE_C_FLAGS}\")\n"
  "else()\n"
  "    set(CMAKE_C_FLAGS \"-fPIC -fPIE -O3 -Werror -fstack-protector-strong -fomit-frame-pointer ${CMAKE_C_FLAGS}\")\n"
  "    set(CMAKE_C_FLAGS_Release \"${CMAKE_C_FLAGS_Release} -O3 -ffunction-sections -Werror -fdata-sections\")\n"
  "endif()\n"
  "\n"
  "function(create_library)\n"
  "    add_custom_command(TARGET ${PROJ_NAME}\n"
  "            POST_BUILD\n"
  "            COMMAND rm -rf tmp\n"
  "            COMMAND mkdir tmp\n"
  "            COMMAND cd tmp && ar -x ${OP_LIB}\n"
  "            COMMAND echo \"raw static library ${library_name} size:\"\n"
  "            COMMAND ls -lh ${library_name}\n"
  "            COMMAND mv ${library_name} ./tmp && cd tmp && ar -x ${library_name}\n"
  "            COMMENT \"unzip raw static library ${library_name}\"\n"
  "            )\n"
  "    foreach (object_file ${OP_SRC})\n"
  "        add_custom_command(TARGET ${PROJ_NAME} POST_BUILD COMMAND mv ./tmp/${object_file} .)\n"
  "    endforeach ()\n"
  "    add_custom_command(TARGET ${PROJ_NAME}\n"
  "            POST_BUILD\n"
  "            COMMAND ar cr ${library_name} *.o\n"
  "            COMMAND ranlib ${library_name}\n"
  "            COMMAND echo \"new static library ${library_name} size:\"\n"
  "            COMMAND ls -lh ${library_name}\n"
  "            COMMAND rm -rf tmp && rm -rf *.o\n"
  "            COMMENT \"generate specified static library ${library_name}\"\n"
  "            )\n"
  "endfunction(create_library)\n"
  "string(CONCAT library_name \"lib\" ${PROJ_NAME} \".a\")\n"
  "create_library()\n";

#endif  // MICRO_LITE_MICRO_CODER_GENERATOR_CONST_BLOCKS_CMAKE_LISTS_CODE_H_
