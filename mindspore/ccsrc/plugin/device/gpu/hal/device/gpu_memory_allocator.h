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

#ifndef MINDSPORE_CCSRC_RUNTIME_DEVICE_GPU_GPU_MEMORY_ALLOCATOR_H_
#define MINDSPORE_CCSRC_RUNTIME_DEVICE_GPU_GPU_MEMORY_ALLOCATOR_H_

#include <memory>
#include "plugin/device/gpu/hal/device/cuda_driver.h"
#include "backend/common/mem_reuse/mem_dynamic_allocator.h"

namespace mindspore {
namespace device {
namespace gpu {
class GPUMemoryAllocator : public DynamicMemPoolBestFit {
 public:
  ~GPUMemoryAllocator() override = default;
  bool Init();
  void CheckMaxDeviceMemory() const;
  bool AllocBufferQueueMem(size_t size, DeviceMemPtr *addr);

  size_t AllocDeviceMem(size_t size, DeviceMemPtr *addr) override;
  bool FreeDeviceMem(const DeviceMemPtr &addr) override;
  size_t free_mem_size() override;
  size_t AlignMemorySize(size_t size) const override;

  static GPUMemoryAllocator &GetInstance() {
    static GPUMemoryAllocator instance;
    return instance;
  }

 private:
  GPUMemoryAllocator() = default;
  GPUMemoryAllocator(const GPUMemoryAllocator &) = delete;
  GPUMemoryAllocator &operator=(const GPUMemoryAllocator &) = delete;

  // Used to track address of data buffer queue.
  DeviceMemPtr buffer_q_addr_{nullptr};

  float limited_device_memory_{0.0};
  size_t total_used_device_memory_{0};
  size_t available_device_memory_{0};
};
}  // namespace gpu
}  // namespace device
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_DEVICE_GPU_GPU_MEMORY_ALLOCATOR_H_
