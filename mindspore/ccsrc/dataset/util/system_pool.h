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
#ifndef DATASET_UTIL_SYSTEM_POOL_H_
#define DATASET_UTIL_SYSTEM_POOL_H_

#include <cstddef>
#include <cstdlib>
#include <limits>
#include <memory>
#include <new>
#include "./securec.h"
#include "dataset/util/allocator.h"
#include "dataset/util/memory_pool.h"

namespace mindspore {
namespace dataset {
// This class demonstrate how to implement a simple MemoryPool
// for minddata/dataset using malloc/free/realloc.  We need to
// implement 4 virtual functions. Other MemoryPool
// implementation, e.g., are BuddyArena and CircularPool.  All
// these MemoryPool can be used together with Allocator.h for
// C++ STL containers.
class SystemPool : public MemoryPool {
 public:
  ~SystemPool() override {}

  Status Allocate(size_t n, void **pp) override { return DeMalloc(n, pp, false); }

  void Deallocate(void *p) override { free(p); }

  Status Reallocate(void **p, size_t old_sz, size_t new_sz) override {
    if (old_sz >= new_sz) {
      // Do nothing if we shrink.
      return Status::OK();
    } else {
      void *ptr = *p;
      void *q = nullptr;
      RETURN_IF_NOT_OK(DeMalloc(new_sz, &q, false));
      errno_t err = memcpy_s(q, new_sz, ptr, old_sz);
      if (err) {
        free(q);
        RETURN_STATUS_UNEXPECTED(std::to_string(err));
      }
      free(ptr);
      *p = q;
      return Status::OK();
    }
  }

  uint64_t get_max_size() const override { return std::numeric_limits<uint64_t>::max(); }

  int PercentFree() const override { return 100; }

  template <typename T>
  static Allocator<T> GetAllocator() {
    return Allocator<T>(std::make_shared<SystemPool>());
  }
};
}  // namespace dataset
}  // namespace mindspore

#endif  // DATASET_UTIL_SYSTEM_POOL_H_
