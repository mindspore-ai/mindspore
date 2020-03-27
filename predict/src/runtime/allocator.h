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

#ifndef PREDICT_SRC_RUNTIME_ALLOCATOR_H_
#define PREDICT_SRC_RUNTIME_ALLOCATOR_H_

#include <memory>
#include <string>
#include <vector>
#include <mutex>
#include "common/module_registry.h"

namespace mindspore {
namespace predict {
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
  std::vector<MemBuf *> allocatedList;
  std::vector<MemBuf *> freeList;
  int shiftFactor = 0;
  bool lockFlag = false;
};

// these declaration are for module integration, refer to sample_allocator
const char MODULE_REG_NAME_ALLOCATOR[] = "allocator";

template <> class Module<Allocator> : public ModuleBase {
 public:
  virtual std::shared_ptr<Allocator> Create() = 0;
};
}  // namespace predict
}  // namespace mindspore

#endif  // PREDICT_SRC_RUNTIME_ALLOCATOR_H_
