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

#ifndef MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_ASCEND_MEMORY_MANAGER_H_
#define MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_ASCEND_MEMORY_MANAGER_H_
#include "runtime/device/memory_manager.h"
#include "graphengine/inc/external/runtime/rt_error_codes.h"
namespace mindspore {
namespace device {
namespace ascend {
class AscendMemoryManager : public MemoryManager {
 public:
  AscendMemoryManager() = default;
  ~AscendMemoryManager() override = default;

  void MallocDeviceMemory() override;
  void FreeDeviceMemory() override;
  void ResetDynamicMemory() override;
  void ClearGlobalIdleMem() override;
  void *MallocMemFromMemPool(size_t size) override;
  uint64_t GetDeviceMemSize();
  void MallocSomasDynamicMem(const session::KernelGraph *graph);
  uint8_t *MallocCommunicationMemFromMemPool(size_t size) override;

 protected:
  uint8_t *MallocStaticMem(size_t size, bool communication_mem, uint32_t graph_id = kInvalidGraphId) override;
  uint8_t *MallocDynamicMem(size_t size, bool communication_mem) override;

 private:
  uint8_t *device_mem_pool_base_{nullptr};
  uint64_t device_mem_pool_size_{0};

  uint64_t GetDeviceMemSizeFromContext();
};
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_ASCEND_MEMORY_MANAGER_H_
