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
#ifndef MINDSPORE_CCSRC_RUNTIME_HARDWARE_CPU_CPU_DEVICE_CONTEXT_H_
#define MINDSPORE_CCSRC_RUNTIME_HARDWARE_CPU_CPU_DEVICE_CONTEXT_H_

#include <vector>
#include <memory>
#include "runtime/hardware/device_context.h"
#include "runtime/hardware/device_context_manager.h"
#include "runtime/device/memory_manager.h"

namespace mindspore {
namespace device {
namespace cpu {
class CPUDeviceContext : public DeviceContext {
 public:
  explicit CPUDeviceContext(const DeviceContextKey &device_context_key)
      : DeviceContext(device_context_key), mem_manager_(nullptr), initialized_(false) {}
  ~CPUDeviceContext() override = default;

  bool Initialize() override;

  bool AllocateMemory(DeviceAddress *const &address, size_t size) const override;
  void FreeMemory(DeviceAddress *const &address) const override;

  void SetOperatorInfo(const std::vector<CNodePtr> &nodes) const override;
  void CreateKernel(const std::vector<CNodePtr> &nodes) const override;

  bool LaunchKernel(KernelMod *kernel_mod, const std::vector<AddressPtr> &inputs,
                    const std::vector<AddressPtr> &workspace, const std::vector<AddressPtr> &outputs) const override;

 private:
  DISABLE_COPY_AND_ASSIGN(CPUDeviceContext);

  std::shared_ptr<MemoryManager> mem_manager_;
  bool initialized_;
};
}  // namespace cpu
}  // namespace device
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_HARDWARE_CPU_CPU_DEVICE_CONTEXT_H_
