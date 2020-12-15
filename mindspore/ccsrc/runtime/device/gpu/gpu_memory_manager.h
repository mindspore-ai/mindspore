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

#ifndef MINDSPORE_CCSRC_RUNTIME_DEVICE_GPU_GPU_MEMORY_MANAGER_H_
#define MINDSPORE_CCSRC_RUNTIME_DEVICE_GPU_GPU_MEMORY_MANAGER_H_
#include <vector>
#include "runtime/device/memory_manager.h"
namespace mindspore {
namespace device {
namespace gpu {
class GPUMemoryManager : public MemoryManager {
 public:
  GPUMemoryManager() = default;
  virtual ~GPUMemoryManager() = default;

  void MallocDeviceMemory() override;
  void FreeDeviceMemory() override;

  void *MallocMemFromMemPool(size_t size) override;
  void FreeMemFromMemPool(void *device_ptr) override;
  std::vector<void *> MallocContinuousMemFromMemPool(size_t total_size, std::vector<size_t> size_list) override;
  bool MallocContinuousMemFromMemPool(const DeviceAddressPtrList addr_list, size_t total_size,
                                      std::vector<size_t> size_list) override;

 protected:
  uint8_t *MallocStaticMem(size_t size, bool communication_mem, uint32_t graph_id = kInvalidGraphId) override;
};
}  // namespace gpu
}  // namespace device
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_RUNTIME_DEVICE_GPU_GPU_MEMORY_MANAGER_H_
