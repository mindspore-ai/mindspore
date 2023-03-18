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

#include <vector>
#include "runtime/device/memory_manager.h"
#include "plugin/device/ascend/hal/device/ascend_memory_pool.h"

namespace mindspore {
namespace device {
namespace ascend {
class AscendMemoryManager : public MemoryManager {
 public:
  AscendMemoryManager() = default;
  ~AscendMemoryManager() override = default;

  void Initialize() override;
  void Finalize() override;
  void ResetDynamicMemory() override;
  void ClearGlobalIdleMem() override;
  void *MallocMemFromMemPool(size_t size, bool from_persistent_mem) override;
  void *MallocOverflowMemFromMemFromMemPool(size_t size, bool from_persistent_mem);
  void FreeMemFromMemPool(void *device_ptr) override;
  uint64_t GetMsMaxMemSize() const;
  uint8_t *MallocCommunicationMemFromMemPool(size_t size) override;
  bool MallocContinuousMemFromMemPool(const DeviceAddressPtrList &addr_list, size_t total_size,
                                      std::vector<size_t> size_list) override;
  std::vector<void *> MallocContinuousMemFromMemPool(const std::vector<size_t> &size_list) override {
    return AscendMemoryPool::GetInstance().AllocContinuousTensorMem(size_list);
  }

  void SwapIn(const void *host_ptr, void *device_ptr, size_t mem_size, void *stream) override;
  void SwapOut(const void *device_ptr, void *host_ptr, size_t mem_size, void *stream) override;
  size_t GetAvailableMemSize() override;
  uint64_t GetMsUsedHbmSize() const;

 protected:
  uint8_t *MallocStaticMem(size_t size, bool communication_mem, uint32_t graph_id) override;
  uint8_t *MallocDynamicMem(size_t size, bool communication_mem) override;
};
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_ASCEND_MEMORY_MANAGER_H_
