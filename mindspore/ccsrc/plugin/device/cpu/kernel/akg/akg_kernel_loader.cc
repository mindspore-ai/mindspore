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

#include "plugin/device/cpu/kernel/akg/akg_kernel_loader.h"
#include <sys/mman.h>
#include <unistd.h>
#include <dlfcn.h>
#include <elf.h>
#include <cstring>
#include <cerrno>
#include <iostream>

#ifdef BUILD_LITE
#include "src/common/log_adapter.h"
#else
#include "utils/log_adapter.h"
#endif

namespace mindspore {
namespace kernel {
static const std::vector<std::string> rodata_section_list = {".rodata", ".rodata.cst4", ".rodata.cst8", ".rodata.cst16",
                                                             ".rodata.cst32"};
constexpr const size_t got_table_item_size = 8;
bool AkgLibraryLoader::LoadAkgLib(const void *data) {
  obj_.base = static_cast<const uint8_t *>(data);
  return ParseObj();
}

size_t AkgLibraryLoader::PageAlign(size_t n) const { return (n + (page_size_ - 1)) & ~(page_size_ - 1); }

const Elf64_Shdr *AkgLibraryLoader::LookupSection(const std::string &name) const {
  size_t name_len = name.size();
  /* number of entries in the section table is encoded in the ELF header */
  for (Elf64_Half i = 0; i < obj_.hdr->e_shnum; i++) {
    const char *section_name = shstrtab_ + sections_[i].sh_name;
    size_t section_name_len = strlen(section_name);
    if (name_len == section_name_len && strcmp(name.data(), section_name) == 0) {
      /* ignore section with 0 size */
      if (sections_[i].sh_size != 0) {
        return sections_ + i;
      }
    }
  }
  return nullptr;
}

void *AkgLibraryLoader::LookupFunction(const std::string &name) const {
  size_t name_len = name.size();
  /* loop through all the symbols_ in the symbol table and
   * find the offset of the function from the
   * beginning of the .text section */
  for (size_t i = 0; i < num_symbols_; i++) {
    /* consider only function symbols_ */
    if (ELF64_ST_TYPE(symbols_[i].st_info) == STT_FUNC) {
      const char *function_name = strtab_ + symbols_[i].st_name;
      size_t function_name_len = strlen(function_name);
      if (name_len == function_name_len && strcmp(name.data(), function_name) == 0) {
        return text_runtime_base_ + symbols_[i].st_value;
      }
    }
  }
  return nullptr;
}

std::vector<uint64_t> AkgLibraryLoader::ParseSection(std::string section_name, bool is_mandatory) const {
  const Elf64_Shdr *section_hdr = LookupSection(section_name);
  uint64_t section_size = 0;
  uint64_t section_offset = 0;
  if (!section_hdr) {
    if (is_mandatory) {
      MS_LOG(ERROR) << "Failed to find" << section_name;
    } else {
      MS_LOG(INFO) << "Failed to find" << section_name;
    }
  } else {
    section_size = section_hdr->sh_size;
    section_offset = section_hdr->sh_offset;
  }
  return {section_size, section_offset};
}

bool AkgLibraryLoader::ParseObj(void) {
  /* the section table offset is encoded in the ELF header */
  sections_ = static_cast<const Elf64_Shdr *>(static_cast<const void *>(obj_.base + obj_.hdr->e_shoff));
  shstrtab_ =
    static_cast<const char *>(static_cast<const void *>(obj_.base + sections_[obj_.hdr->e_shstrndx].sh_offset));
  /* find the .symtab entry in the section table */
  const Elf64_Shdr *symtab_hdr = LookupSection(".symtab");
  if (!symtab_hdr) {
    MS_LOG(ERROR) << ("Failed to find .symtab");
    return false;
  }
  /* the symbols_ table */
  symbols_ = static_cast<const Elf64_Sym *>(static_cast<const void *>(obj_.base + symtab_hdr->sh_offset));
  num_symbols_ = symtab_hdr->sh_size / symtab_hdr->sh_entsize;
  const Elf64_Shdr *strtab_hdr = LookupSection(".strtab");
  if (!strtab_hdr) {
    MS_LOG(ERROR) << ("Failed to find .strtab");
    return false;
  }
  strtab_ = static_cast<const char *>(static_cast<const void *>(obj_.base + strtab_hdr->sh_offset));
  page_size_ = static_cast<size_t>(sysconf(static_cast<int>(_SC_PAGESIZE)));
  /* find the .text entry in the section table */
  auto text_vec = ParseSection(".text", true);
  if (text_vec.empty()) {
    return false;
  }
  /* find the .data section in the section table */
  auto data_vec = ParseSection(".data");
  /* find the .data or .rodata.xxx section in the section table */
  size_t const_section_size = 0;
  std::vector<uint64_t> page_align_list;
  (void)page_align_list.emplace_back(PageAlign(data_vec[0]));
  for (auto &item : rodata_section_list) {
    auto section_vec = ParseSection(item);
    const_section_size += PageAlign(section_vec[0]);
    (void)page_align_list.emplace_back(PageAlign(section_vec[0]));
    section_info_map_[item] = section_vec;
  }
  CountDynmaicSymbols();
  /* allocate memory for .text, .data , .rodata.xxx copies , jumptable for external symbols and gottable , rounding
   * up each section to whole pages. */
  mmap_size_ = PageAlign(text_vec[0]) + const_section_size + PageAlign(sizeof(struct ext_jump) * num_ext_symbols_) +
               PageAlign(got_table_size_);
  text_runtime_base_ =
    static_cast<u_int8_t *>(mmap(nullptr, mmap_size_, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
  if (static_cast<void *>(text_runtime_base_) == MAP_FAILED) {
    MS_LOG(ERROR) << "Failed to allocate memory in akg kernel loader, mmap size = " << mmap_size_
                  << " , errno: " << errno << ".";
    return false;
  }
  /*  will come after .data */
  if (text_vec[0] != 0 && memcpy_s(text_runtime_base_, text_vec[0], obj_.base + text_vec[1], text_vec[0]) != EOK) {
    MS_LOG(ERROR) << "memcpy .text section failed, section size:" << text_vec[0];
    return false;
  }
  /* .data will come right after .text */
  data_runtime_base_ = text_runtime_base_ + PageAlign(text_vec[0]);
  if (data_vec[0] != 0 && memcpy_s(data_runtime_base_, data_vec[0], obj_.base + data_vec[1], data_vec[0]) != EOK) {
    MS_LOG(ERROR) << "memcpy .data section failed, section size:" << data_vec[0];
    return false;
  }
  /* .rodata.xxx will come right after .data */
  auto rodata_runtime_base = data_runtime_base_ + PageAlign(data_vec[0]);
  uint8_t *current_rodata_runtime_base = rodata_runtime_base;
  for (size_t idx = 0; idx < rodata_section_list.size(); idx++) {
    section_runtime_base_map_[rodata_section_list[idx]] = current_rodata_runtime_base + page_align_list[idx];
    auto section_size = section_info_map_[rodata_section_list[idx]][0];
    auto section_offset = section_info_map_[rodata_section_list[idx]][1];
    if (section_size != 0 &&
        memcpy_s(current_rodata_runtime_base, section_size, obj_.base + section_offset, section_size) != EOK) {
      MS_LOG(ERROR) << "memcpy " << rodata_section_list[idx] << "section failed, section size:" << section_size;
      return false;
    }
  }
  /* jumptable_ will come at last */
  jumptable_ = static_cast<ext_jump *>(static_cast<void *>(current_rodata_runtime_base + page_align_list.back()));
  got_table_ =
    static_cast<uint8_t *>(static_cast<void *>(jumptable_ + PageAlign(sizeof(struct ext_jump) * num_ext_symbols_)));
  auto ret = DoTextRelocations();
  if (!ret) {
    return ret;
  }
  /* make the .text copy readonly and executable */
  if (mprotect(text_runtime_base_, PageAlign(text_vec[0]), PROT_READ | PROT_EXEC) != 0) {
    MS_LOG(ERROR) << "Failed to make .text executable";
    return false;
  }
  /* make the .rodata.xxx copy readonly */
  if (mprotect(rodata_runtime_base, static_cast<size_t>(current_rodata_runtime_base - rodata_runtime_base),
               PROT_READ) != 0) {
    MS_LOG(ERROR) << "Failed to make .rodata readonly";
    return false;
  }
  /* make the jumptable readonly and executable */
  if (mprotect(jumptable_, PageAlign(sizeof(ext_jump) * num_ext_symbols_), PROT_READ | PROT_EXEC) != 0) {
    MS_LOG(ERROR) << "Failed to make the jumptable executable";
    return false;
  }
  /* make the gottable readonly */
  if (mprotect(got_table_, PageAlign(got_table_size_), PROT_READ) != 0) {
    MS_LOG(ERROR) << "Failed to make the jumptable executable";
    return false;
  }
  return true;
}

void *AkgLibraryLoader::LookupExtFunction(const std::string &name) {
  lib_handle_ = dlopen(nullptr, RTLD_LAZY | RTLD_DEEPBIND);
  if (lib_handle_ == nullptr) {
    MS_LOG(ERROR) << "dlopen handler is nullptr.";
    return nullptr;
  }
  return dlsym(lib_handle_, name.c_str());
}

void AkgLibraryLoader::CountDynmaicSymbols() {
  const Elf64_Shdr *rela_text_hdr = LookupSection(".rela.text");
  if (!rela_text_hdr) {
    MS_LOG(INFO) << "cannot find .rela.text, no need to do relocation.";
    return;
  }
  size_t num_relocations = rela_text_hdr->sh_size / rela_text_hdr->sh_entsize;
  const Elf64_Rela *relocations =
    static_cast<const Elf64_Rela *>(static_cast<const void *>(obj_.base + rela_text_hdr->sh_offset));
  for (size_t i = 0; i < num_relocations; i++) {
    auto symbol_idx = ELF64_R_SYM(relocations[i].r_info);
    auto type = ELF64_R_TYPE(relocations[i].r_info);
    if (symbols_[symbol_idx].st_shndx == SHN_UNDEF) {
      /* external symbol reference */
      num_ext_symbols_++;
    } else if (type == R_X86_64_GOTPCREL || type == R_X86_64_GOTPCREL64) {
      /* x86_64 signed PC relative offset to GOT */
      got_table_size_ += got_table_item_size;
    } else if (type == R_AARCH64_ADR_GOT_PAGE) {
      /* aarch64 signed PC relative offset to GOT */
      got_table_size_ += got_table_item_size;
    }
  }
}

uint8_t *AkgLibraryLoader::SectionRuntimeBase(const Elf64_Shdr *section) {
  const char *section_name = shstrtab_ + section->sh_name;
  size_t section_name_len = strlen(section_name);
  std::string section_name_str = std::string(section_name);
  if (strlen(".text") == section_name_len && strcmp(".text", section_name) == 0) {
    return text_runtime_base_;
  }
  if (strlen(".data") == section_name_len && strcmp(".data", section_name) == 0) {
    return data_runtime_base_;
  }
  if (section_runtime_base_map_.find(section_name_str) != section_runtime_base_map_.end()) {
    return section_runtime_base_map_.at(section_name_str);
  }
  MS_LOG(ERROR) << "No runtime base address for section " << section_name;
  return nullptr;
}

void *AkgLibraryLoader::RelocateExtSymbols(size_t symbol_idx, size_t jump_table_idx) {
  /* genergate instructions for external symbol/function by name and return jumptable instr address */
  auto addr = LookupExtFunction(strtab_ + symbols_[symbol_idx].st_name);
  if (addr == nullptr) {
    MS_LOG(ERROR) << "cannot find " << strtab_ + symbols_[symbol_idx].st_name << " in mindspore library.";
    return nullptr;
  }
#if defined(ENABLE_ARM64) || defined(__aarch64__)
  /* arm64 branch instrution.
   * For details, please check
   * https://developer.arm.com/documentation/ddi0596/2021-12/Base-Instructions/MOVK--Move-wide-with-keep-?lang=en
   * The generated instructions in jumptable always like:
   * assume 'addr = 0x0000ffffad688138'
   * mov x17,#0x8138
   * movk x17,#0xad68, lsl #16
   * movk x17,#0xffff, lsl #32
   * movk x17,#0x0, lsl #48
   * br x17
   */
  int idx = 0;
  uint64_t func_addr = reinterpret_cast<uint64_t>(addr);
  uint32_t movk_x17 = 0xF2800011;
  constexpr const uint32_t addr_shift = 0x10;
  constexpr const uint32_t mask = 0xFFFF;
  constexpr const uint32_t hw_shift_size = 21;
  constexpr const uint32_t br_x17 = 0xD61F0220;
  constexpr const int shift_size = 4;
  constexpr const int addr_mask_shift_size = 5;
  for (int shift = 0; shift < shift_size;) {
    auto masked_imm16_shift = (func_addr & mask) << addr_mask_shift_size;
    if (idx == 0) {
      movk_x17 = 0xD2800011;
    }
    uint32_t hw = shift << hw_shift_size;
    jumptable_[jump_table_idx].instr[idx] = movk_x17 ^ hw ^ masked_imm16_shift;
    idx++;
    shift++;
    func_addr = func_addr >> addr_shift;
  }
  jumptable_[jump_table_idx].instr[idx] = br_x17;
#elif defined(__x86_64__)
  /* unconditional x64 JMP instruction which jump to addr in ext_jump
   * JMP instruction should always be {0xff, 0x25, 0xf2, 0xff, 0xff, 0xff}
   */
  constexpr const uint8_t jmp_instr[] = {0xff, 0x25, 0xf2, 0xff, 0xff, 0xff};
  for (size_t j = 0; j < sizeof(jmp_instr) / sizeof(uint8_t); j++) {
    jumptable_[jump_table_idx].instr[j] = jmp_instr[j];
  }
  jumptable_[jump_table_idx].addr = static_cast<u_int8_t *>(addr);
#else
  MS_LOG(ERROR) << "Akg kernel loader only support aarch64 and x86_64.";
  return nullptr;
#endif
  return static_cast<void *>(&(jumptable_[jump_table_idx].instr));
}

bool AkgLibraryLoader::DoTextRelocations() {
  const Elf64_Shdr *rela_text_hdr = LookupSection(".rela.text");
  if (!rela_text_hdr) {
    /* no need to do relocation. */
    return true;
  }
  size_t num_relocations = rela_text_hdr->sh_size / rela_text_hdr->sh_entsize;
  const Elf64_Rela *relocations =
    static_cast<const Elf64_Rela *>(static_cast<const void *>(obj_.base + rela_text_hdr->sh_offset));
  size_t got_table_offset = 0;
  size_t jump_table_idx = 0;
  for (size_t i = 0; i < num_relocations; i++) {
    auto symbol_idx = ELF64_R_SYM(relocations[i].r_info);
    auto type = ELF64_R_TYPE(relocations[i].r_info);
    /* where to patch .text */
    void *patch_offset = text_runtime_base_ + relocations[i].r_offset;
    /* symbol address with respect to which the relocation is performed */
    void *symbol_address = nullptr;
    auto ndx = symbols_[symbol_idx].st_shndx;
    /* if this is an external symbol */
    switch (ndx) {
      case SHN_UNDEF: {
        symbol_address = RelocateExtSymbols(symbol_idx, jump_table_idx);
        jump_table_idx++;
        break;
      }
      default: {
        symbol_address = SectionRuntimeBase(&sections_[symbols_[symbol_idx].st_shndx]) + symbols_[symbol_idx].st_value;
      }
    }
    if (!symbol_address) {
      MS_LOG(ERROR) << "Can not find " << strtab_ + symbols_[symbol_idx].st_name << " symbol address.";
      return false;
    }

    switch (type) {
      case R_AARCH64_ADR_GOT_PAGE: {
        /* P-page-rel using adrp instruction */
        constexpr const uint64_t page_bits = 0xFFFFFFFFFFFFF000;
        constexpr const uint64_t distance_shift = 12;
        constexpr const uint64_t immhi_shift = 5;
        constexpr const uint64_t immlo_shift = 29;
        auto base = reinterpret_cast<uint64_t>(patch_offset) & page_bits;
        auto got_table_page = reinterpret_cast<uint64_t>(got_table_ + got_table_offset) & (~(page_size_ - 1));
        auto distance = (got_table_page - base) >> distance_shift;
        /* low 2 bits in distance */
        auto immlo = distance & ((1 << 2) - 1);
        /* high 18 bits in distance */
        auto immhi = (distance >> ((1 << 2) - 1)) & ((1 << 17) - 1);
        /* patch the adrp instruction */
        *(static_cast<uint32_t *>(patch_offset)) |=
          (static_cast<uint32_t>(immhi << immhi_shift) | static_cast<uint32_t>(immlo << immlo_shift));
        *(static_cast<uint64_t *>(static_cast<void *>(got_table_ + got_table_offset))) =
          reinterpret_cast<uint64_t>(symbol_address);
        break;
      }
      case R_AARCH64_LD64_GOT_LO12_NC: {
        /* Patch the ldr instruction */
        constexpr const uint32_t page_offset_shift = 10;
        auto page_offset = reinterpret_cast<uint64_t>(got_table_ + got_table_offset) & (page_size_ - 1);
        *(static_cast<uint32_t *>(patch_offset)) |= static_cast<uint32_t>(page_offset << page_offset_shift);
        got_table_offset += got_table_item_size;
        break;
      }
      case R_AARCH64_CALL26: {
        /* This can jump to +/-128MB. We are able to this because jumptable is close to .text */
        constexpr const uint32_t aarch64_call_addr_shift = 2;
        *(static_cast<uint32_t *>(patch_offset)) ^=
          (static_cast<uint32_t>((static_cast<uint8_t *>(symbol_address) + relocations[i].r_addend -
                                  static_cast<uint8_t *>(patch_offset))) >>
           aarch64_call_addr_shift);
        break;
      }
      case R_X86_64_GOTPCREL:
      /* 32 bit signed PC relative offset to GOT , which follows G + GOT + A - P */
      case R_X86_64_GOTPCREL64:
        /* 64 bit signed PC relative offset to GOT , which follows G + GOT + A - P */
        *(static_cast<uint32_t *>(patch_offset)) =
          got_table_ + got_table_offset + relocations[i].r_addend - static_cast<uint8_t *>(patch_offset);
        *(static_cast<uint64_t *>(static_cast<void *>(got_table_ + got_table_offset))) =
          reinterpret_cast<uint64_t>(symbol_address);
        got_table_offset += got_table_item_size;
        break;
      case R_X86_64_PC32:
        /*  For .rodata.xxx symbols */
      case R_X86_64_PLT32: {
        /*  Change 'call 0000' into 'call &jumptable' for external symbols */
        *(static_cast<uint32_t *>(patch_offset)) = static_cast<uint32_t>(
          static_cast<uint8_t *>(symbol_address) + relocations[i].r_addend - static_cast<uint8_t *>(patch_offset));
        break;
      }
      default: {
        MS_LOG(ERROR) << "Can not support relocation type:" << type
                      << ", whose symbol is : " << strtab_ + symbols_[symbol_idx].st_name;
        return false;
      }
    }
  }
  return true;
}

AkgLibraryLoader::~AkgLibraryLoader() {
  if (lib_handle_) {
    (void)dlclose(lib_handle_);
  }
  if (text_runtime_base_ != nullptr) {
    (void)munmap(text_runtime_base_, mmap_size_);
  }
}
}  // namespace kernel
}  // namespace mindspore
