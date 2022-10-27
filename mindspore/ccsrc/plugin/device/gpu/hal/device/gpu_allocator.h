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
#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_HAL_DEVICE_GPU_ALLOCATOR_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_HAL_DEVICE_GPU_ALLOCATOR_H_

#include "plugin/device/gpu/hal/device/gpu_memory_allocator.h"

namespace mindspore {
namespace device {
namespace gpu {
// A general Allocator used to allocate GPU memory.
template <typename T>
class GPUAllocator {
 public:
  using value_type = T;

  GPUAllocator() = default;
  ~GPUAllocator() = default;

  template <typename U>
  GPUAllocator(const GPUAllocator<U> &other) {}

  template <typename U>
  GPUAllocator &operator=(const GPUAllocator<U> &other) {
    return *this;
  }

  template <typename U>
  GPUAllocator(GPUAllocator<U> &&other) {}

  template <typename U>
  GPUAllocator &operator=(GPUAllocator<U> &&other) {
    return *this;
  }

  // Allocate GPU memory from dynamic memory pool.
  // The name of the allocate function cannot be changed and is used to call std::allocator_traits::allocate.
  value_type *allocate(std::size_t n) {
    auto ptr = GPUMemoryAllocator::GetInstance().AllocTensorMem(sizeof(value_type) * n);
    return reinterpret_cast<value_type *>(ptr);
  }

  // Free GPU memory.
  // The name of the deallocate function cannot be changed and is used to call std::allocator_traits::deallocate.
  void deallocate(value_type *ptr, std::size_t) {
    GPUMemoryAllocator::GetInstance().FreeTensorMem(static_cast<void *>(ptr));
  }
};

template <typename T, typename U>
bool operator==(GPUAllocator<T> const &, GPUAllocator<U> const &) noexcept {
  return true;
}

template <typename T, typename U>
bool operator!=(GPUAllocator<T> const &lhs, GPUAllocator<U> const &rhs) noexcept {
  return false;
}
}  // namespace gpu
}  // namespace device
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_HAL_DEVICE_GPU_ALLOCATOR_H_
