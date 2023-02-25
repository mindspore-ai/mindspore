/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
void CodeCMakeNetLibrary(std::ofstream &ofs, const std::unique_ptr<CoderContext> &ctx, const Configurator *config) {
  ofs << "include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../include/)\n";
  ofs << "include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../)\n";
  if (config->target() == kCortex_M) {
    ofs << "include_directories(${OP_HEADER_PATH}/CMSIS/NN/Include)\n"
        << "include_directories(${OP_HEADER_PATH}/CMSIS/DSP/Include)\n"
        << "include_directories(${OP_HEADER_PATH}/CMSIS/Core/Include)\n";
  }
  ofs << "set(OP_SRC\n";
  for (const std::string &c_file : ctx->c_files()) {
    ofs << "    " << c_file << ".o\n";
  }
  for (int i = 0; i <= ctx->GetCurModelIndex(); ++i) {
    ofs << "    weight" << i << ".c.o\n"
        << "    net" << i << ".c.o\n"
        << "    model" << i << ".c.o\n";
  }
  ofs << "    model.c.o\n"
      << "    context.c.o\n"
      << "    tensor.c.o\n";
  if (config->target() != kCortex_M) {
    ofs << "    allocator.c.o\n";
  }
  if (config->debug_mode()) {
    ofs << "    debug_utils.c.o\n";
  }
  if (config->support_parallel()) {
    ofs << "    micro_core_affinity.c.o\n"
           "    micro_thread_pool.c.o\n";
  }
  ofs << ")\n";
  std::set<std::string> kernel_cmake_asm_set_files = ctx->asm_files();
  if (!kernel_cmake_asm_set_files.empty() && (config->target() == kARM32 || config->target() == kARM64)) {
    ofs << "set(ASSEMBLY_SRC\n";
    for (const std::string &asm_file : kernel_cmake_asm_set_files) {
      ofs << "    " << asm_file << ".o\n";
    }
    ofs << ")\n"
        << "set_property(SOURCE ${ASSEMBLY_SRC} PROPERTY LANGUAGE C)\n"
        << "list(APPEND OP_SRC ${ASSEMBLY_SRC})\n";
  }
  ofs << "file(GLOB_RECURSE NET_SRC\n"
         "     ${CMAKE_CURRENT_SOURCE_DIR}/*.cc\n"
         "     ${CMAKE_CURRENT_SOURCE_DIR}/*.c\n"
         "     )\n"
         "add_library(net STATIC ${NET_SRC})\n";
}
}  // namespace mindspore::lite::micro
