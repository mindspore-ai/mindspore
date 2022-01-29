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

#include "src/custom_allocator.h"
#include <utility>
#include "include/svp_acl.h"
#include "include/svp_acl_ext.h"
#include "src/custom_log.h"
#include "src/common_utils.h"

namespace mindspore {
namespace dpico {
CustomAllocator::CustomAllocator(size_t aligned_size) { aligned_size_ = aligned_size; }

CustomAllocator::~CustomAllocator() { Clear(); }

void CustomAllocator::SetContext(const AllocatorContext &ctx) {
  lockFlag_ = ctx.lockFlag;
  shiftFactor_ = static_cast<unsigned>(ctx.shiftFactor);
}

void CustomAllocator::Lock() {
  if (lockFlag_) {
    lock_.lock();
  }
}

void CustomAllocator::UnLock() {
  if (lockFlag_) {
    lock_.unlock();
  }
}

bool CustomAllocator::ReuseMemory(size_t free_size, size_t size) const {
  return free_size >= size &&
         (free_size <= (size >= UINT32_MAX / (1ul << shiftFactor_) ? UINT32_MAX : size << shiftFactor_));
}

void *CustomAllocator::Malloc(size_t size) {
  if (size > lite::GetMaxMallocSize()) {
    MS_LOG(ERROR) << "MallocData out of max_size, size: " << size;
    return nullptr;
  }
  if (this->total_size_ >= lite::GetMaxMallocSize()) {
    MS_LOG(ERROR) << "Memory pool is exhausted";
    return nullptr;
  }
  Lock();
  auto iter = freeList_.lower_bound(size);
  if (iter != freeList_.end() && ReuseMemory(iter->second->size, size)) {
    auto membuf = iter->second;
    membuf->ref_count_ = 0;
    (void)freeList_.erase(iter);
    allocatedList_[membuf->buf] = membuf;
    UnLock();
    return membuf->buf;
  }

  void *mem_ptr = nullptr;
  svp_acl_error ret =
    svp_acl_rt_malloc_cached(&mem_ptr, sizeof(MemBuf) + size + aligned_size_, SVP_ACL_MEM_MALLOC_NORMAL_ONLY);
  if (ret != SVP_ACL_SUCCESS) {
    MS_LOG(ERROR) << "malloc data failed.";
    return nullptr;
  }
  std::unique_ptr<MemBuf> membuf(reinterpret_cast<MemBuf *>(mem_ptr));
  if (membuf == nullptr) {
    MS_LOG(ERROR) << "malloc membuf return nullptr";
    svp_acl_rt_free(mem_ptr);
    UnLock();
    return nullptr;
  }
  this->total_size_ += size;
  membuf->ref_count_ = 0;
  membuf->size = size;
  membuf->buf = reinterpret_cast<void *>(
    (reinterpret_cast<uintptr_t>(membuf.get()) + sizeof(MemBuf) + aligned_size_ - 1) & (~(aligned_size_ - 1)));
  auto bufPtr = membuf->buf;
  allocatedList_[bufPtr] = membuf.release();
  UnLock();
  return bufPtr;
}

void CustomAllocator::Free(void *buf) {
  if (buf == nullptr) {
    return;
  }
  Lock();
  auto iter = allocatedList_.find(buf);
  if (iter != allocatedList_.end()) {
    auto membuf = iter->second;
    membuf->ref_count_ = 0;
    (void)allocatedList_.erase(iter);
    (void)freeList_.insert(std::make_pair(membuf->size, membuf));
    UnLock();
    return;
  }
  UnLock();
  svp_acl_rt_free(buf);
}

int CustomAllocator::RefCount(void *buf) {
  if (buf == nullptr) {
    return -1;
  }
  Lock();
  auto iter = allocatedList_.find(buf);
  if (iter != allocatedList_.end()) {
    auto membuf = iter->second;
    int ref_count = std::atomic_load(&membuf->ref_count_);
    UnLock();
    return ref_count;
  }
  UnLock();
  return -1;
}
int CustomAllocator::SetRefCount(void *buf, int ref_count) {
  if (buf == nullptr) {
    return -1;
  }
  Lock();
  auto iter = allocatedList_.find(buf);
  if (iter != allocatedList_.end()) {
    auto membuf = iter->second;
    std::atomic_store(&membuf->ref_count_, ref_count);
    UnLock();
    return ref_count;
  }
  UnLock();
  return -1;
}
int CustomAllocator::IncRefCount(void *buf, int ref_count) {
  if (buf == nullptr) {
    return -1;
  }
  Lock();
  auto iter = allocatedList_.find(buf);
  if (iter != allocatedList_.end()) {
    auto membuf = iter->second;
    auto ref = std::atomic_fetch_add(&membuf->ref_count_, ref_count);
    UnLock();
    return (ref + ref_count);
  }
  UnLock();
  return -1;
}
int CustomAllocator::DecRefCount(void *buf, int ref_count) {
  if (buf == nullptr) {
    return -1;
  }
  Lock();
  auto iter = allocatedList_.find(buf);
  if (iter != allocatedList_.end()) {
    auto membuf = iter->second;
    auto ref = std::atomic_fetch_sub(&membuf->ref_count_, ref_count);
    UnLock();
    return (ref - ref_count);
  }
  UnLock();
  return -1;
}
void CustomAllocator::Clear() {
  Lock();

  for (auto &it : allocatedList_) {
    svp_acl_rt_free(it.second);
  }
  allocatedList_.clear();

  for (auto &it : freeList_) {
    svp_acl_rt_free(it.second);
  }
  freeList_.clear();
  UnLock();
}
}  // namespace dpico
}  // namespace mindspore
