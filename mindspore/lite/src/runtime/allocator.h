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

namespace mindspore::lite {
struct AllocatorContext {
  int shiftFactor;
  bool lockFlag;
};

class Allocator {
 public:
  Allocator() : name("default") {}
  virtual ~Allocator() {}
  virtual void *Malloc(size_t size) = 0;
  virtual void Free(void *ptr) = 0;
  virtual void SetContext(const AllocatorContext &ctx) {}
  virtual size_t GetTotalSize() { return 0; }
  virtual void Clear() {}
  static std::shared_ptr<Allocator> Create();
  std::string name;
};

class DefaultAllocator : public Allocator {
 public:
  DefaultAllocator();
  ~DefaultAllocator() override;
  void SetContext(const AllocatorContext &ctx) override;
  void *Malloc(size_t size) override;
  void Free(void *ptr) override;
  size_t GetTotalSize() override;
  void Clear() override;

 private:
  void Lock();
  void UnLock();
  struct MemBuf {
    size_t size;
    void *buf;
  };

  std::mutex lock;
  // <membuf->buf, membuf>
  std::unordered_map<void *, MemBuf *> allocatedList;
  std::multimap<size_t, MemBuf *> freeList;
  // 6 is empirical value
  int shiftFactor = 6;
  bool lockFlag = false;
};

#define MAX_MALLOC_SIZE 500 * 1024 * 1024

}  // namespace mindspore::lite

#endif  // MINDSPORE_LITE_SRC_RUNTIME_ALLOCATOR_H_

