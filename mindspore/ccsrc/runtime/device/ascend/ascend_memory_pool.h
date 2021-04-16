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

#ifndef MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_ASCEND_MEMORY_POOL_H_
#define MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_ASCEND_MEMORY_POOL_H_

#include <memory>
#include "backend/optimizer/mem_reuse/mem_dynamic_allocator.h"

namespace mindspore {
namespace device {
namespace ascend {
class AscendMemoryPool : public DynamicMemPoolBestFit {
 public:
  ~AscendMemoryPool() override = default;
  AscendMemoryPool(const AscendMemoryPool &) = delete;
  AscendMemoryPool &operator=(const AscendMemoryPool &) = delete;

  void Init(uint8_t *device_mem_base, uint64_t device_mem_size, uint64_t dynamic_mem_offset);
  size_t AllocDeviceMem(size_t size, DeviceMemPtr *addr) override;
  bool FreeDeviceMem(const DeviceMemPtr &addr) override;
  void ResetIdleMemBuf();
  void set_device_mem_size(uint64_t device_mem_size);
  void set_device_mem_pool_base(uint8_t *device_mem_pool_base);
  void set_device_mem_pool_offset(uint64_t device_mem_pool_offset);
  void set_graph_dynamic_mem_offset(uint64_t graph_dynamic_mem_offset);

  uint64_t device_mem_pool_offset() const;
  size_t free_mem_size() override;
  size_t total_mem_size() override;

  static AscendMemoryPool &GetInstance() {
    static AscendMemoryPool instance;
    return instance;
  }

 protected:
  // The real size by memory alloc aligned.
  size_t AlignMemorySize(size_t size) const override;
  // Calculate memory block required alloc size when adding the memory block.
  size_t CalMemBlockAllocSize(size_t size) override;

 private:
  AscendMemoryPool() = default;
  uint8_t *device_mem_pool_base_{nullptr};
  uint64_t device_mem_size_{0};
  uint64_t device_mem_pool_offset_{0};
  uint64_t graph_dynamic_mem_offset_{0};
};
}  // namespace ascend
}  // namespace device
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_ASCEND_MEMORY_POOL_H_
