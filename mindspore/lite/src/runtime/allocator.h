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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_ALLOCATOR_H_
#define MINDSPORE_LITE_SRC_RUNTIME_ALLOCATOR_H_

#include <memory>
#include <string>
#include <vector>
#include <mutex>
#include <map>
#include <unordered_map>
#include <unordered_set>

namespace mindspore {

struct AllocatorContext {
  int shiftFactor;
  bool lockFlag;
};

class Allocator {
 public:
  Allocator() : name("default") {}
  virtual ~Allocator() = default;
  virtual void *Malloc(size_t size) = 0;
  virtual void Free(void *ptr) = 0;
  virtual void SetContext(const AllocatorContext &ctx) {}
  virtual size_t total_size() = 0;
  static std::shared_ptr<Allocator> Create();
  virtual void *Prepare(void *ptr) { return ptr; }
  std::string name;
};

class DefaultAllocator : public Allocator {
 public:
  DefaultAllocator();
  ~DefaultAllocator() override;
  void SetContext(const AllocatorContext &ctx) override;
  void *Malloc(size_t size) override;
  void Free(void *ptr) override;
  size_t total_size() override { return this->total_size_; }
  void Clear();

 private:
  void Lock();
  void UnLock();
  struct MemBuf {
    size_t size;
    void *buf;
  };

  std::mutex lock_;
  size_t total_size_ = 0;
  // <membuf->buf, membuf>
  std::unordered_map<void *, MemBuf *> allocatedList_;
  std::multimap<size_t, MemBuf *> freeList_;
  // 6 is empirical value
  int shiftFactor_ = 6;
  bool lockFlag_ = false;
};

constexpr int64_t MAX_MALLOC_SIZE = static_cast<size_t>(2000) * 1024 * 1024;
constexpr int64_t MAX_THREAD_POOL_SIZE = static_cast<size_t>(3000) * 1024 * 1024;

}  // namespace mindspore

#endif  // MINDSPORE_LITE_SRC_RUNTIME_ALLOCATOR_H_
