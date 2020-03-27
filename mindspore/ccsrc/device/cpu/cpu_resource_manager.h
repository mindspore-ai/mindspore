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
#ifndef MINDSPORE_CCSRC_DEVICE_CPU_CPU_RESOURCE_MANAGER_H_
#define MINDSPORE_CCSRC_DEVICE_CPU_CPU_RESOURCE_MANAGER_H_

#include <vector>
#include <unordered_map>
#include "session/kernel_graph.h"
#include "device/device_address.h"
#include "device/cpu/cpu_simple_mem_plan.h"
namespace mindspore {
namespace device {
namespace cpu {
class CPUResourceManager {
 public:
  CPUResourceManager() = default;
  ~CPUResourceManager();

  void MemPlan(const session::KernelGraph *graph);
  void MemMalloc(const session::KernelGraph *graph);
  void ResetAddressRefCount(const session::KernelGraph *graph);
  void DecreaseAddressRefCount(const AnfNodePtr &kernel);
  void *MemMalloc(size_t mem_size);
  void MemFree(void *ptr);

 private:
  void MemFree();
  CPUSimpleMemPlan mem_plan_;

  size_t mem_size_{0};
  uint8_t *mem_ptr_{nullptr};
  bool dynamic_malloc_{false};
  std::unordered_map<void *, size_t> dynamic_mem_;
};
}  // namespace cpu
}  // namespace device
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_DEVICE_CPU_CPU_RESOURCE_MANAGER_H_
