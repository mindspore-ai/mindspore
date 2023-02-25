/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "coder/generator/component/allocator_component.h"
#include <memory>
#include <utility>
#include <algorithm>
#include "coder/generator/component/const_blocks/license.h"
#include "coder/utils/coder_utils.h"
#include "coder/opcoders/parallel.h"

namespace mindspore::lite::micro {
void CodeAllocatorFileHeader(std::ofstream &ofs) {
  ofs << g_hwLicense;
  ofs << "#ifndef MINDSPORE_LITE_MICRO_LIBRARY_SOURCE_ALLOCATOR_H_\n"
         "#define MINDSPORE_LITE_MICRO_LIBRARY_SOURCE_ALLOCATOR_H_\n\n"
      << "#include <stddef.h>\n"
      << "#include \"stdatomic.h\"\n"
      << "#include \"stdbool.h\"\n";
  ofs << R"RAW(
typedef struct MemBlock {
#ifdef __clang__
  atomic_bool occupied;
#else
  bool occupied;
#endif
  size_t size;
  void *addr;
  struct MemBlock *next;
} MemBlock;

void IncRefCount();
void DecRefCount();
void *GlobalMemory();
void *Malloc(size_t size);
bool LockBuffer(void *block);
bool UnLockBuffer(void *block);
)RAW";
  ofs << "\n#endif  // MINDSPORE_LITE_MICRO_LIBRARY_SOURCE_ALLOCATOR_H_\n";
}

void CodeCalcRefCount(std::ofstream &ofs) {
  ofs << R"RAW(
void IncRefCount() { ++kReferenceCount; }

void DecRefCount() {
  --kReferenceCount;
  bool expected = false;
  while (!atomic_compare_exchange_strong(&kLock, &expected, true)) {
    expected = false;
  }
  if (kReferenceCount == 0) {
    while (mem_block != NULL) {
      MemBlock *next = mem_block->next;
      free(mem_block);
      mem_block = next;
    }
  }
  kLock = false;
}

)RAW";
}

void CodeGlobalMemory(std::ofstream &ofs, size_t size) {
  ofs << "void *GlobalMemory() {\n"
      << "size_t init_size = " << size << ";\n";
  ofs << R"RAW(
  bool expected = false;
  while (!atomic_compare_exchange_strong(&kLock, &expected, true)) {
    expected = false;
  }
  if (mem_block != NULL) {
    kLock = false;
    return mem_block;
  }
  mem_block = malloc(sizeof(MemBlock) + init_size);
  mem_block->occupied = false;
  mem_block->size = init_size;
  mem_block->addr = (char *)mem_block + sizeof(MemBlock);
  mem_block->next = NULL;
  kLock = false;
  return mem_block;
}
)RAW";
}

void CodeMemoryOccupied(std::ofstream &ofs) {
  ofs << R"RAW(
void *Malloc(size_t size) {
  bool expected = false;
  while (!atomic_compare_exchange_strong(&kLock, &expected, true)) {
    expected = false;
  }
  if (mem_block == NULL) {
    kLock = false;
    return NULL;
  }
  MemBlock *pre = mem_block;
  MemBlock *cur = mem_block;
  MemBlock *find = NULL;
  while (cur != NULL) {
    if (cur->size < size) {
      break;
    }
    if (!cur->occupied) {
      find = cur;
    }
    pre = cur;
    cur = cur->next;
  }
  if (find != NULL) {
    find->occupied = true;
    kLock = false;
    return find;
  }
  MemBlock *block = malloc(sizeof(MemBlock) + size);
  block->occupied = true;
  block->size = size;
  block->addr = (char *)block + sizeof(MemBlock);
  block->next = NULL;
  block->next = pre->next;
  pre->next = block;
  kLock = false;
  return block;
}
)RAW";
}

void CodeLockOccupied(std::ofstream &ofs) {
  ofs << R"RAW(
bool LockBuffer(void *block) {
  MemBlock *m_block = block;
  bool expected = false;
  return atomic_compare_exchange_strong(&(m_block->occupied), &expected, true);
}

bool UnLockBuffer(void *block) {
  MemBlock *m_block = block;
  bool expected = true;
  return atomic_compare_exchange_strong(&(m_block->occupied), &expected, false);
}
)RAW";
}
}  // namespace mindspore::lite::micro
