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

#ifndef MINDSPORE_CCSRC_DEVICE_ASCEND_ASCEND_MEMORY_POOL_H_
#define MINDSPORE_CCSRC_DEVICE_ASCEND_ASCEND_MEMORY_POOL_H_

#include <memory>
#include "pre_activate/mem_reuse/mem_dynamic_allocator.h"

namespace mindspore {
namespace device {
namespace ascend {
class AscendMemoryPool : public DynamicMemPoolBestFit {
 public:
  ~AscendMemoryPool() override = default;
  AscendMemoryPool(const AscendMemoryPool&) = delete;
  AscendMemoryPool& operator=(const AscendMemoryPool&) = delete;

  size_t AllocDeviceMem(size_t size, DeviceMemPtr* addr) override;
  bool FreeDeviceMem(const DeviceMemPtr& addr) override;
  void set_device_mem_pool_base(uint8_t* device_mem_pool_base);
  void set_device_mem_pool_size(uint64_t device_mem_pool_size) {
    device_mem_pool_size_ = device_mem_pool_size;
    free_mem_size_ = device_mem_pool_size_;
    total_mem_size_ = free_mem_size_;
  }
  size_t free_mem_size() override;
  size_t total_mem_size() override;

  static AscendMemoryPool& GetInstance() {
    static AscendMemoryPool instance;
    return instance;
  }

 protected:
  // The real size by memory alloc aligned.
  size_t AlignMemorySize(size_t size) const override;
  // Get the minimum memory unit size using for dynamic extend.
  size_t mem_alloc_unit_size() const override;

 private:
  AscendMemoryPool() = default;
  bool has_malloc_{false};
  uint8_t* device_mem_pool_base_{nullptr};
  uint64_t device_mem_pool_size_{0};
  size_t free_mem_size_{0};
  size_t total_mem_size_{0};
};
}  // namespace ascend
}  // namespace device
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_DEVICE_ASCEND_ASCEND_MEMORY_POOL_H_
