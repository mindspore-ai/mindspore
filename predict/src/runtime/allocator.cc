/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#include "src/runtime/allocator.h"
#include "common/module_registry.h"
#include "src/op_common.h"

namespace mindspore {
namespace predict {
std::shared_ptr<Allocator> Allocator::Create() {
  auto alloc = GetRegistryInstance()->Create<Allocator>(MODULE_REG_NAME_ALLOCATOR);
  if (alloc != nullptr) {
    return alloc;
  }

  // default allocator
  return std::shared_ptr<Allocator>(new DefaultAllocator());
}

DefaultAllocator::DefaultAllocator() = default;

DefaultAllocator::~DefaultAllocator() { Clear(); }

void DefaultAllocator::SetContext(const AllocatorContext &ctx) {
  lockFlag = ctx.lockFlag;
  shiftFactor = ctx.shiftFactor;
}

void DefaultAllocator::Lock() {
  if (lockFlag) {
    lock.lock();
  }
}

void DefaultAllocator::UnLock() {
  if (lockFlag) {
    lock.unlock();
  }
}

void *DefaultAllocator::Malloc(size_t size) {
  if (size > MAX_MALLOC_SIZE) {
    return nullptr;
  }
  Lock();
  auto it = freeList.begin();
  for (; it != freeList.end(); it++) {
    auto membuf = *it;

    if ((membuf->size >= size) && (membuf->size < (size << shiftFactor))) {
      freeList.erase(it);
      allocatedList.push_back(membuf);
      UnLock();
      return membuf->buf;
    }
  }
  std::unique_ptr<MemBuf> membuf(reinterpret_cast<MemBuf *>(malloc(sizeof(MemBuf) + size)));
  if (membuf == nullptr) {
    UnLock();
    return nullptr;
  }
  membuf->size = size;
  membuf->buf = reinterpret_cast<char *>(membuf.get()) + sizeof(MemBuf);
  auto bufPtr = membuf->buf;
  allocatedList.push_back(membuf.release());
  UnLock();
  return bufPtr;
}

void DefaultAllocator::Free(void *buf) {
  if (buf == nullptr) {
    return;
  }
  Lock();
  auto it = allocatedList.begin();
  for (; it != allocatedList.end(); it++) {
    auto membuf = *it;

    if (membuf->buf == buf) {
      allocatedList.erase(it);
      freeList.push_back(membuf);
      UnLock();
      return;
    }
  }
  UnLock();
  free(buf);
}

size_t DefaultAllocator::GetTotalSize() {
  Lock();
  size_t totalSize = 0;
  auto it = allocatedList.begin();
  for (; it != allocatedList.end(); it++) {
    auto membuf = *it;
    totalSize += membuf->size;
  }
  it = freeList.begin();
  for (; it != freeList.end(); it++) {
    auto membuf = *it;
    totalSize += membuf->size;
  }
  UnLock();
  return totalSize;
}

void DefaultAllocator::Clear() {
  Lock();
  auto it = allocatedList.begin();
  for (; it != allocatedList.end(); it++) {
    free(*it);
  }
  allocatedList.clear();
  it = freeList.begin();
  for (; it != freeList.end(); it++) {
    free(*it);
  }
  freeList.clear();
  UnLock();
}
}  // namespace predict
}  // namespace mindspore
