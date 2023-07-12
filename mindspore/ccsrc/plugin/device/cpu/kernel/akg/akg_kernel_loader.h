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
#ifndef MINDSPORE_LITE_TOOLS_GRAPH_KERNEL_RUNTIME_AKG_KERNEL_LOADER_H_
#define MINDSPORE_LITE_TOOLS_GRAPH_KERNEL_RUNTIME_AKG_KERNEL_LOADER_H_
#include <elf.h>

#include <string>
#include <vector>
#include <map>
#include "plugin/device/cpu/kernel/akg/akg_kernel_loader.h"

namespace mindspore {
namespace kernel {
class AkgLibraryLoader {
 public:
  bool LoadAkgLib(const void *data);
  void *LookupFunction(const std::string &name) const;
  ~AkgLibraryLoader();

 private:
  const Elf64_Shdr *LookupSection(const std::string &name) const;
  bool ParseObj();
  void *LookupExtFunction(const std::string &name);
  void CountDynmaicSymbols();
  uint8_t *SectionRuntimeBase(const Elf64_Shdr *section);
  bool DoTextRelocations();
  void *RelocateExtSymbols(size_t symbol_idx, size_t jump_table_idx);
  std::vector<uint64_t> ParseSection(std::string section_name, bool is_mandatory = false) const;
  uint64_t PageAlign(uint64_t n) const;
  using objhdr = union {
    const Elf64_Ehdr *hdr;
    const uint8_t *base;
  };
  objhdr obj_;
  /* sections table pointer */
  const Elf64_Shdr *sections_;
  const char *shstrtab_ = nullptr;
  /* symbols table pointer */
  const Elf64_Sym *symbols_;
  /* number of entries in the symbols table */
  size_t num_symbols_;
  /* string table pointer */
  const char *strtab_ = nullptr;
  size_t page_size_;
  /* runtime base address of the imported code */
  uint8_t *text_runtime_base_ = nullptr;
  /* runtime base of the .data section */
  uint8_t *data_runtime_base_ = nullptr;
  /* .rodata section info */
  std::map<const std::string, std::vector<uint64_t>> section_info_map_;
  std::map<const std::string, uint8_t *> section_runtime_base_map_;
  /* number of external symbols in the symbol table */
  size_t num_ext_symbols_ = 0;
  void *lib_handle_ = nullptr;
  size_t mmap_size_ = 0;
  struct ext_jump {
#if defined(ENABLE_ARM64) || defined(__aarch64__)
    uint32_t instr[5];
#elif defined(__x86_64__)
    uint8_t *addr;
    uint8_t instr[6];
#else
#endif
  };
  struct ext_jump *jumptable_;
  /* got table */
  uint8_t *got_table_;
  size_t got_table_size_ = 0;
};
}  // namespace kernel
}  // namespace mindspore
#endif
