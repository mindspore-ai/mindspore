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

  GPUAllocator() : use_memory_pool_(true) {}
  explicit GPUAllocator(bool use_memory_pool) : use_memory_pool_(use_memory_pool) {}
  ~GPUAllocator() = default;

  template <typename U>
  GPUAllocator(const GPUAllocator<U> &other) {
    this->use_memory_pool_ = other.use_memory_pool_;
  }

  template <typename U>
  GPUAllocator &operator=(const GPUAllocator<U> &other) {
    this->use_memory_pool_ = other.use_memory_pool_;
    return *this;
  }

  template <typename U>
  GPUAllocator(GPUAllocator<U> &&other) {
    this->use_memory_pool_ = other.use_memory_pool_;
  }

  template <typename U>
  GPUAllocator &operator=(GPUAllocator<U> &&other) {
    this->use_memory_pool_ = other.use_memory_pool_;
    return *this;
  }

  // Allocate GPU memory.
  // The name of the allocate function cannot be changed and is used to call std::allocator_traits::allocate.
  value_type *allocate(std::size_t n) {
    if (use_memory_pool_) {
      auto ptr = GPUMemoryAllocator::GetInstance().AllocTensorMem(sizeof(value_type) * n);
      MS_EXCEPTION_IF_NULL(ptr);
      return reinterpret_cast<value_type *>(ptr);
    }

    static std::mutex mutex;
    std::unique_lock<std::mutex> lock(mutex);
    void *ptr = nullptr;
    auto ret = cudaMalloc(reinterpret_cast<void **>(&ptr), sizeof(value_type) * n);
    if (ret != cudaSuccess) {
      MS_LOG(EXCEPTION) << "Call cudaMalloc failed, ret[" << static_cast<int>(ret) << "], " << cudaGetErrorString(ret);
    }
    MS_EXCEPTION_IF_NULL(ptr);
    return reinterpret_cast<value_type *>(ptr);
  }

  // Free GPU memory.
  // The name of the deallocate function cannot be changed and is used to call std::allocator_traits::deallocate.
  void deallocate(value_type *ptr, std::size_t) {
    if (use_memory_pool_) {
      GPUMemoryAllocator::GetInstance().FreeTensorMem(ptr);
      return;
    }

    auto ret = cudaFree(ptr);
    if (ret != cudaSuccess) {
      MS_LOG(EXCEPTION) << "Call cudaFree failed, ret[" << static_cast<int>(ret) << "], " << cudaGetErrorString(ret);
    }
  }

  // Whether use gpu memory pool to allocate/deallocate memory.
  bool use_memory_pool_;
};

template <typename T, typename U>
bool operator==(GPUAllocator<T> const &lhs, GPUAllocator<U> const &rhs) noexcept {
  return lhs.use_memory_pool_ == rhs.use_memory_pool_;
}

template <typename T, typename U>
bool operator!=(GPUAllocator<T> const &lhs, GPUAllocator<U> const &rhs) noexcept {
  return lhs.use_memory_pool_ != rhs.use_memory_pool_;
}
}  // namespace gpu
}  // namespace device
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_HAL_DEVICE_GPU_ALLOCATOR_H_
