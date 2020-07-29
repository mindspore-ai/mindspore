/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include <utility>
#include "utils/log_adapter.h"

namespace mindspore::lite {
std::shared_ptr<Allocator> Allocator::Create() { return std::shared_ptr<Allocator>(new DefaultAllocator()); }

DefaultAllocator::DefaultAllocator() {}

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
    MS_LOG(ERROR) << "MallocData out of max_size, size: " << size;
    return nullptr;
  }
  Lock();
  auto iter = freeList.lower_bound(size);
  if (iter != freeList.end() && (iter->second->size >= size) && (iter->second->size < (size << shiftFactor))) {
    auto membuf = iter->second;
    freeList.erase(iter);
    allocatedList[membuf->buf] = membuf;
    UnLock();
    return membuf->buf;
  }

  std::unique_ptr<MemBuf> membuf(reinterpret_cast<MemBuf *>(malloc(sizeof(MemBuf) + size)));
  if (membuf == nullptr) {
    MS_LOG(ERROR) << "malloc membuf return nullptr";
    UnLock();
    return nullptr;
  }
  membuf->size = size;
  membuf->buf = reinterpret_cast<char *>(membuf.get()) + sizeof(MemBuf);
  auto bufPtr = membuf->buf;
  allocatedList[bufPtr] = membuf.release();
  UnLock();
  return bufPtr;
}

void DefaultAllocator::Free(void *buf) {
  if (buf == nullptr) {
    return;
  }
  Lock();
  auto iter = allocatedList.find(buf);
  if (iter != allocatedList.end()) {
    auto membuf = iter->second;
    allocatedList.erase(iter);
    freeList.insert(std::make_pair(membuf->size, membuf));
    UnLock();
    return;
  }
  UnLock();
  free(buf);
}

size_t DefaultAllocator::GetTotalSize() {
  Lock();
  size_t totalSize = 0;

  for (auto it = allocatedList.begin(); it != allocatedList.end(); it++) {
    auto membuf = it->second;
    totalSize += membuf->size;
  }

  for (auto it = freeList.begin(); it != freeList.end(); it++) {
    auto membuf = it->second;
    totalSize += membuf->size;
  }
  UnLock();
  return totalSize;
}

void DefaultAllocator::Clear() {
  Lock();

  for (auto it = allocatedList.begin(); it != allocatedList.end(); it++) {
    free(it->second);
  }
  allocatedList.clear();

  for (auto it = freeList.begin(); it != freeList.end(); it++) {
    free(it->second);
  }
  freeList.clear();
  UnLock();
}
}  // namespace mindspore::lite

