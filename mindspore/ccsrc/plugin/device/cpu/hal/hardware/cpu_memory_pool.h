/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_RUNTIME_HARDWARE_CPU_CPU_MEMORY_POOL_H_
#define MINDSPORE_CCSRC_RUNTIME_HARDWARE_CPU_CPU_MEMORY_POOL_H_

#include <memory>
#include "utils/ms_utils.h"
#include "backend/common/mem_reuse/mem_dynamic_allocator.h"

namespace mindspore {
namespace device {
namespace cpu {
class BACKEND_EXPORT CPUMemoryPool : public DynamicMemPoolBestFit {
 public:
  ~CPUMemoryPool() override = default;

  static CPUMemoryPool &GetInstance() {
    static CPUMemoryPool instance;
    return instance;
  }

  size_t AllocDeviceMem(size_t size, DeviceMemPtr *addr) override;
  bool FreeDeviceMem(const DeviceMemPtr &addr) override;
  size_t free_mem_size() override;

 private:
  CPUMemoryPool() = default;
  DISABLE_COPY_AND_ASSIGN(CPUMemoryPool);

  size_t total_used_memory_{0};
};
}  // namespace cpu
}  // namespace device
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_HARDWARE_CPU_CPU_MEMORY_POOL_H_
