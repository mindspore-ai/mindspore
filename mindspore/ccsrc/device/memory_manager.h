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

#ifndef MINDSPORE_MINDSPORE_CCSRC_DEVICE_MEMORY_MANAGER_H_
#define MINDSPORE_MINDSPORE_CCSRC_DEVICE_MEMORY_MANAGER_H_
#include <memory>
#include "pre_activate/mem_reuse/mem_reuse.h"
#include "pre_activate/mem_reuse/mem_reuse_allocator.h"
namespace mindspore {
namespace device {
const int kStaticMem = 0;
const int kDynamicMem = 1;
const int kReuseDynamicMem = 2;
const int kGetAllOuts = -1;
const uint64_t kMemAlignSize = 512;
using MemReuseUtilPtr = mindspore::memreuse::MemReuseUtilPtr;

class MemoryManager {
 public:
  MemoryManager() = default;
  virtual ~MemoryManager() = default;

  virtual void MallocDeviceMemory() = 0;
  virtual void FreeDeviceMemory() = 0;
  void ResetDynamicMemory() {
    total_dynamic_size_ = 0;
    dynamic_mem_offset_ = 0;
  }

  void MallocReusedDynamicMem(session::KernelGraph *graph);
  uint8_t *MallocOutputMem(const AnfNodePtr &node, size_t index, int flag, size_t size);
  uint8_t *MallocWorkSpaceMem(const AnfNodePtr &node, size_t index, int flag, size_t size);
  virtual uint8_t *MallocMem(int flag, size_t size);

  virtual void MallocMemFromMemPool(const DeviceAddressPtr address, size_t size);
  virtual void *MallocMemFromMemPool(size_t size);
  virtual void FreeMemFromMemPool(void *device_ptr);

  size_t GetCommonAlignSize(size_t input_size) const;
  size_t GetCommunicationAlignSize(size_t input_size) const;

 protected:
  virtual uint8_t *MallocStaticMem(size_t size, bool communication_mem);
  virtual uint8_t *MallocDynamicMem(size_t size, bool communication_mem);
  uint8_t *device_mem_base_{nullptr};
  uint64_t device_mem_size_{0};
  uint64_t dynamic_mem_offset_{0};
  uint64_t static_mem_offset_{0};
  size_t total_static_size_ = 0;
  size_t total_dynamic_size_ = 0;
  MemReuseUtilPtr mem_reuse_util_ptr_{nullptr};
};
}  // namespace device
}  // namespace mindspore
#endif  // MINDSPORE_MINDSPORE_CCSRC_DEVICE_MEMORY_MANAGER_H_
