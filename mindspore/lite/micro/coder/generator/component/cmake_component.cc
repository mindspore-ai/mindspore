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

#include "coder/generator/component/cmake_component.h"
#include <set>
#include <memory>

namespace mindspore::lite::micro {

void CodeCMakeNetLibrary(std::ofstream &ofs, const std::string &module_name, const std::unique_ptr<CoderContext> &ctx,
                         Target target) {
  ofs << "include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../include/)\n";
  if (target == kARM32M) {
    ofs << "include_directories(${OP_HEADER_PATH}/cmsis)\n"
        << "include_directories(${OP_HEADER_PATH}/cmsis/CMSIS/NN/Include)\n"
        << "include_directories(${OP_HEADER_PATH}/cmsis/CMSIS/DSP/Include)\n"
        << "include_directories(${OP_HEADER_PATH}/cmsis/CMSIS/Core/Include)\n";
  }
  ofs << "set(OP_SRC\n";
  for (const std::string &c_file : ctx->c_files()) {
    ofs << "    " << c_file << ".o\n";
  }
  ofs << "    " << module_name << "_weight.c.o\n"
      << "    " << module_name << ".c.o\n"
      << ")\n";

  std::set<std::string> kernel_cmake_asm_set_files = ctx->asm_files();
  if (!kernel_cmake_asm_set_files.empty()) {
    ofs << "set(ASSEMBLY_SRC\n";
    for (const std::string &asm_file : kernel_cmake_asm_set_files) {
      ofs << "    " << asm_file << ".o\n";
    }
    ofs << ")\n"
        << "set_property(SOURCE ${ASSEMBLY_SRC} PROPERTY LANGUAGE C)\n"
        << "list(APPEND OP_SRC ${ASSEMBLY_SRC})\n";
  }

  ofs << "file(GLOB NET_SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.c)\n"
      << "add_library(${PROJ_NAME} STATIC ${NET_SRC})\n";
}

}  // namespace mindspore::lite::micro
