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
#ifndef MINDSPORE_CCSRC_DISTRIBUTED_EMBEDDING_CACHE_ALLOCATOR_H_
#define MINDSPORE_CCSRC_DISTRIBUTED_EMBEDDING_CACHE_ALLOCATOR_H_

#include "runtime/hardware/device_context_manager.h"

namespace mindspore {
namespace distributed {
// A general Allocator used to allocate host memory.
template <typename T>
class Allocator {
 public:
  using value_type = T;

  Allocator() {
    device::DeviceContextKey host_key = {"CPU", 0};
    cpu_device_context_ = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(host_key);
    MS_EXCEPTION_IF_NULL(cpu_device_context_);
    cpu_device_context_->Initialize();
  }

  ~Allocator() = default;

  template <typename U>
  Allocator(const Allocator<U> &other) {
    cpu_device_context_ = other.cpu_device_context_;
  }

  template <typename U>
  Allocator &operator=(const Allocator<U> &other) {
    cpu_device_context_ = other.cpu_device_context_;
    return *this;
  }

  template <typename U>
  Allocator(Allocator<U> &&other) {
    cpu_device_context_ = other.cpu_device_context_;
    other.cpu_device_context_ = nullptr;
  }

  template <typename U>
  Allocator &operator=(Allocator<U> &&other) {
    cpu_device_context_ = other.cpu_device_context_;
    other.cpu_device_context_ = nullptr;
    return *this;
  }

  // Allocate host memory from dynamic memory pool.
  // Parameter[in] `n`: The number of value_type object to allocate for memory. If sizeof(value_type) is 1, the n
  // indicates the bytes to allocate for memory.
  // Return the pointer to the allocated memory.
  // Note that the name of the allocate function cannot be changed and is used to call std::allocator_traits::allocate.
  value_type *allocate(std::size_t n) {
    MS_EXCEPTION_IF_NULL(cpu_device_context_);
    MS_EXCEPTION_IF_NULL(cpu_device_context_->device_res_manager_);
    void *ptr = cpu_device_context_->device_res_manager_->AllocateMemory(sizeof(value_type) * n);
    return reinterpret_cast<value_type *>(ptr);
  }

  // Free host memory to dynamic memory pool.
  // Parameter[in] `ptr`: The pointer need to free.
  // Note that the name of the deallocate function cannot be changed and is used to call
  // std::allocator_traits::deallocate.
  void deallocate(value_type *ptr, std::size_t) {
    MS_EXCEPTION_IF_NULL(cpu_device_context_);
    MS_EXCEPTION_IF_NULL(cpu_device_context_->device_res_manager_);
    cpu_device_context_->device_res_manager_->FreeMemory(ptr);
  }

 private:
  // The CPU device context used to allocate host memory.
  device::DeviceContext *cpu_device_context_;
};

template <typename T, typename U>
bool operator==(Allocator<T> const &lhs, Allocator<U> const &rhs) noexcept {
  return lhs.cpu_device_context_ == rhs.cpu_device_context_;
}

template <typename T, typename U>
bool operator!=(Allocator<T> const &lhs, Allocator<U> const &rhs) noexcept {
  return lhs.cpu_device_context_ != rhs.cpu_device_context_;
}
}  // namespace distributed
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_DISTRIBUTED_EMBEDDING_CACHE_ALLOCATOR_H_
