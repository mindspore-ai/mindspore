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
#include "src/common/log_adapter.h"

namespace mindspore {
std::shared_ptr<Allocator> Allocator::Create() {
  return std::shared_ptr<Allocator>(new (std::nothrow) DefaultAllocator());
}

DefaultAllocator::DefaultAllocator() = default;

DefaultAllocator::~DefaultAllocator() { Clear(); }

void DefaultAllocator::SetContext(const AllocatorContext &ctx) {
  lockFlag_ = ctx.lockFlag;
  shiftFactor_ = ctx.shiftFactor;
}

void DefaultAllocator::Lock() {
  if (lockFlag_) {
    lock_.lock();
  }
}

void DefaultAllocator::UnLock() {
  if (lockFlag_) {
    lock_.unlock();
  }
}

void *DefaultAllocator::Malloc(size_t size) {
  if (size > MAX_MALLOC_SIZE) {
    MS_LOG(ERROR) << "MallocData out of max_size, size: " << size;
    return nullptr;
  }
  if (this->total_size_ >= MAX_THREAD_POOL_SIZE) {
    MS_LOG(ERROR) << "Memory pool is exhausted";
    return nullptr;
  }
  Lock();
  auto iter = freeList_.lower_bound(size);
  if (iter != freeList_.end() && (iter->second->size >= size) && (iter->second->size <= (size << shiftFactor_))) {
    auto membuf = iter->second;
    freeList_.erase(iter);
    allocatedList_[membuf->buf] = membuf;
    UnLock();
    return membuf->buf;
  }

  std::unique_ptr<MemBuf> membuf(reinterpret_cast<MemBuf *>(malloc(sizeof(MemBuf) + size)));
  if (membuf == nullptr) {
    MS_LOG(ERROR) << "malloc membuf return nullptr";
    UnLock();
    return nullptr;
  }
  this->total_size_ += size;
  membuf->size = size;
  membuf->buf = reinterpret_cast<char *>(membuf.get()) + sizeof(MemBuf);
  auto bufPtr = membuf->buf;
  allocatedList_[bufPtr] = membuf.release();
  UnLock();
  return bufPtr;
}

void DefaultAllocator::Free(void *buf) {
  if (buf == nullptr) {
    return;
  }
  Lock();
  auto iter = allocatedList_.find(buf);
  if (iter != allocatedList_.end()) {
    auto membuf = iter->second;
    allocatedList_.erase(iter);
    freeList_.insert(std::make_pair(membuf->size, membuf));
    UnLock();
    return;
  }
  UnLock();
  free(buf);
  buf = nullptr;
}

void DefaultAllocator::Clear() {
  Lock();

  for (auto &it : allocatedList_) {
    free(it.second);
  }
  allocatedList_.clear();

  for (auto &it : freeList_) {
    free(it.second);
  }
  freeList_.clear();
  UnLock();
}
}  // namespace mindspore
