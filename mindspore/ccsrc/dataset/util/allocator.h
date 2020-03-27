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
#ifndef DATASET_UTIL_ALLOCATOR_H_
#define DATASET_UTIL_ALLOCATOR_H_

#include <cstdlib>
#include <memory>
#include <type_traits>
#include "dataset/util/memory_pool.h"

namespace mindspore {
namespace dataset {
// The following conforms to the requirements of
// std::allocator. Do not rename/change any needed
// requirements, e.g. function names, typedef etc.
template <typename T>
class Allocator {
 public:
  template <typename U>
  friend class Allocator;

  using value_type = T;
  using pointer = T *;
  using const_pointer = const T *;
  using reference = T &;
  using const_reference = const T &;
  using size_type = uint64_t;

  template <typename U>
  struct rebind {
    using other = Allocator<U>;
  };

  using propagate_on_container_copy_assignment = std::true_type;
  using propagate_on_container_move_assignment = std::true_type;
  using propagate_on_container_swap = std::true_type;

  explicit Allocator(const std::shared_ptr<MemoryPool> &b) : pool_(b) {}

  ~Allocator() = default;

  template <typename U>
  explicit Allocator(Allocator<U> const &rhs) : pool_(rhs.pool_) {}

  template <typename U>
  bool operator==(Allocator<U> const &rhs) const {
    return pool_ == rhs.pool_;
  }

  template <typename U>
  bool operator!=(Allocator<U> const &rhs) const {
    return pool_ != rhs.pool_;
  }

  pointer allocate(std::size_t n) {
    void *p;
    Status rc = pool_->Allocate(n * sizeof(T), &p);
    if (rc.IsOk()) {
      return reinterpret_cast<pointer>(p);
    } else if (rc.IsOutofMemory()) {
      throw std::bad_alloc();
    } else {
      throw std::exception();
    }
  }

  void deallocate(pointer p, std::size_t n = 0) noexcept { pool_->Deallocate(p); }

  size_type max_size() { return pool_->get_max_size(); }

 private:
  std::shared_ptr<MemoryPool> pool_;
};
}  // namespace dataset
}  // namespace mindspore

#endif  // DATASET_UTIL_ALLOCATOR_H_
