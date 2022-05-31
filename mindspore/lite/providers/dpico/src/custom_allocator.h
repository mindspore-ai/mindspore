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

#ifndef MINDSPORE_LITE_PROVIDERS_DPICO_SRC_CUSTOM_ALLOCATOR_H_
#define MINDSPORE_LITE_PROVIDERS_DPICO_SRC_CUSTOM_ALLOCATOR_H_

#include <memory>
#include <string>
#include <vector>
#include <mutex>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <atomic>
#include "include/api/allocator.h"

namespace mindspore {
namespace lite {
struct AllocatorContext {
  int shiftFactor;
  bool lockFlag;
};

class CustomAllocator : public Allocator {
 public:
  explicit CustomAllocator(size_t aligned_size = 32);
  ~CustomAllocator() override;
  void SetContext(const AllocatorContext &ctx);
  void *Malloc(size_t size) override;
  void Free(void *ptr) override;
  int RefCount(void *ptr) override;
  int SetRefCount(void *ptr, int ref_count) override;
  int DecRefCount(void *ptr, int ref_count) override;
  int IncRefCount(void *ptr, int ref_count) override;
  size_t total_size() const { return this->total_size_; }
  void Clear();

 private:
  void Lock();
  void UnLock();
  bool ReuseMemory(size_t free_size, size_t size) const;
  struct MemBuf {
    std::atomic_int ref_count_ = {0};
    size_t size = 0;
    void *buf = nullptr;
  };

  std::mutex lock_;
  size_t total_size_ = 0;
  // <membuf->buf, membuf>
  std::unordered_map<void *, MemBuf *> allocated_list_;
  std::multimap<size_t, MemBuf *> free_list_;
  // 6 is empirical value
  unsigned shift_factor_ = 6;
  bool lock_flag_ = true;
  size_t max_malloc_size_ = 0;
};
using CustomAllocatorPtr = std::shared_ptr<CustomAllocator>;
constexpr int64_t MAX_MALLOC_SIZE = static_cast<size_t>(2000) * 1024 * 1024;
constexpr int64_t MAX_THREAD_POOL_SIZE = static_cast<size_t>(3000) * 1024 * 1024;
}  // namespace lite
}  // namespace mindspore
#endif  // MINDSPORE_LITE_PROVIDERS_DPICO_SRC_CUSTOM_ALLOCATOR_H_
