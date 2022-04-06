/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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
#include <unistd.h>
#include <utility>
#include "include/svp_acl_rt.h"
#include "common/check_base.h"
#include "common/log_util.h"

namespace mindspore {
namespace lite {
namespace {
size_t GetMaxMallocSize() {
  static size_t max_malloc_size =
    static_cast<size_t>(sysconf(_SC_PHYS_PAGES)) * static_cast<size_t>(sysconf(_SC_PAGESIZE));
  return max_malloc_size;
}
}  // namespace
CustomAllocator::CustomAllocator(size_t aligned_size) {
  aligned_size_ = aligned_size;
  max_malloc_size_ = GetMaxMallocSize();
}

CustomAllocator::~CustomAllocator() { Clear(); }

void CustomAllocator::SetContext(const AllocatorContext &ctx) {
  lock_flag_ = ctx.lockFlag;
  shift_factor_ = static_cast<unsigned>(ctx.shiftFactor);
}

void CustomAllocator::Lock() {
  if (lock_flag_) {
    lock_.lock();
  }
}

void CustomAllocator::UnLock() {
  if (lock_flag_) {
    lock_.unlock();
  }
}

bool CustomAllocator::ReuseMemory(size_t free_size, size_t size) const {
  return free_size >= size &&
         (free_size <= (size >= UINT32_MAX / (1ul << shift_factor_) ? UINT32_MAX : size << shift_factor_));
}

void *CustomAllocator::Malloc(size_t size) {
  MS_CHECK_TRUE_MSG(size <= max_malloc_size_, nullptr, "MallocData out of max_size, size: " << size);
  MS_CHECK_TRUE_MSG(total_size_ < max_malloc_size_, nullptr, "memory pool is exhausted");
  Lock();
  auto iter = free_list_.lower_bound(size);
  if (iter != free_list_.end() && ReuseMemory(iter->second->size, size)) {
    auto membuf = iter->second;
    membuf->ref_count_ = 0;
    (void)free_list_.erase(iter);
    allocated_list_[membuf->buf] = membuf;
    UnLock();
    return membuf->buf;
  }
  void *mem_ptr = nullptr;
  int ret = svp_acl_rt_malloc_cached(&mem_ptr, sizeof(MemBuf) + size + aligned_size_, SVP_ACL_MEM_MALLOC_NORMAL_ONLY);
  if (ret != SVP_ACL_SUCCESS) {
    MS_LOG(ERROR) << "svp acl rt malloc cached failed.";
    UnLock();
    return nullptr;
  }
  std::unique_ptr<MemBuf> membuf(reinterpret_cast<MemBuf *>(mem_ptr));
  if (membuf == nullptr) {
    MS_LOG(ERROR) << "malloc membuf return nullptr";
    UnLock();
    return nullptr;
  }
  this->total_size_ += size;
  membuf->ref_count_ = 0;
  membuf->size = size;
  membuf->buf = reinterpret_cast<uint8_t *>(
    (reinterpret_cast<uintptr_t>(membuf.get()) + sizeof(MemBuf) + aligned_size_ - 1) & (~(aligned_size_ - 1)));
  auto buf_ptr = membuf->buf;
  allocated_list_[buf_ptr] = membuf.release();
  UnLock();
  return buf_ptr;
}

void CustomAllocator::Free(void *buf) {
  if (buf == nullptr) {
    return;
  }
  Lock();
  auto iter = allocated_list_.find(buf);
  if (iter != allocated_list_.end()) {
    auto membuf = iter->second;
    membuf->ref_count_ = 0;
    (void)allocated_list_.erase(iter);
    (void)free_list_.insert(std::make_pair(membuf->size, membuf));
    UnLock();
    return;
  }
  UnLock();
  int ret = svp_acl_rt_free(buf);
  MS_CHECK_TRUE_MSG_VOID(ret == SVP_ACL_SUCCESS, "svp acl rt free failed.");
  buf = nullptr;
}

int CustomAllocator::RefCount(void *buf) {
  if (buf == nullptr) {
    return -1;
  }
  Lock();
  auto iter = allocated_list_.find(buf);
  if (iter != allocated_list_.end()) {
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
  auto iter = allocated_list_.find(buf);
  if (iter != allocated_list_.end()) {
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
  auto iter = allocated_list_.find(buf);
  if (iter != allocated_list_.end()) {
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
  auto iter = allocated_list_.find(buf);
  if (iter != allocated_list_.end()) {
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
  int ret;
  for (auto &it : allocated_list_) {
    ret = svp_acl_rt_free(it.second);
    MS_CHECK_TRUE_MSG_VOID(ret == SVP_ACL_SUCCESS, "svp acl rt free failed.");
    it.second = nullptr;
  }
  allocated_list_.clear();

  for (auto &it : free_list_) {
    ret = svp_acl_rt_free(it.second);
    MS_CHECK_TRUE_MSG_VOID(ret == SVP_ACL_SUCCESS, "svp acl rt free failed.");
    it.second = nullptr;
  }
  free_list_.clear();
  UnLock();
}
}  // namespace lite
}  // namespace mindspore
