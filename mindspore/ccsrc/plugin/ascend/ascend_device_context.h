/**
 * Copyright 2022-2025 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_RUNTIME_HARDWARE_ASCEND_ASCEND_DEVICE_CONTEXT_H_
#define MINDSPORE_CCSRC_RUNTIME_HARDWARE_ASCEND_ASCEND_DEVICE_CONTEXT_H_

#include <memory>
#include <string>
#include <map>
#include "include/runtime/hardware_abstract/device_context/device_context.h"
#include "utils/ms_context.h"
#include "plugin/ascend/res_manager/collective/ascend_collective_comm_lib.h"
#include "plugin/ascend/kernel_executor/ascend_kernel_executor.h"
#include "plugin/ascend/res_manager/ascend_res_manager.h"

namespace mindspore {
namespace device {
namespace ascend {
class AscendKernelExecutor;
class AscendResManager;
// The Ascend device properties defined by MindSpore because ACL does not have interface to get this info.
struct AscendDeviceProperties {
  std::string name;
  size_t total_memory;
  size_t free_memory;
};

class AscendDeviceContext : public DeviceInterface<AscendKernelExecutor, AscendResManager> {
 public:
  explicit AscendDeviceContext(const DeviceContextKey &device_context_key) : DeviceInterface(device_context_key) {}
  ~AscendDeviceContext() override = default;

  void Initialize() override;

  void InitializeForAclop() const;

  void Destroy() override;

  static uint32_t GetDeviceCount();
  static std::string GetDeviceName(uint32_t);
  static AscendDeviceProperties GetDeviceProperties(uint32_t);

 private:
  DISABLE_COPY_AND_ASSIGN(AscendDeviceContext);

  void InitDump() const;

  mutable bool initialized_aclop_{false};
  pid_t pid_;  // Indicates the process id which creates the context.
};
}  // namespace ascend
}  // namespace device
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_HARDWARE_ASCEND_ASCEND_DEVICE_CONTEXT_H_
