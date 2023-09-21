/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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
#include <mutex>
#include "runtime/hardware/device_context.h"
#include "runtime/hardware/device_context_manager.h"
#include "plugin/device/ascend/hal/hardware/ascend_device_res_manager.h"
#include "plugin/device/ascend/hal/hardware/ascend_kernel_executor.h"
#include "plugin/device/ascend/hal/hardware/ascend_graph_executor.h"
#include "plugin/device/ascend/hal/hardware/ascend_deprecated_interface.h"
#include "plugin/device/ascend/hal/hardware/ge_kernel_executor.h"

namespace mindspore {
namespace device {
namespace ascend {
class AscendGraphExecutor;
class AscendKernelExecutor;
class AscendDeviceResManager;

class AscendDeviceContext : public DeviceInterface<AscendGraphExecutor, AscendKernelExecutor, AscendDeviceResManager> {
 public:
  explicit AscendDeviceContext(const DeviceContextKey &device_context_key)
      : DeviceInterface(device_context_key), initialized_(false) {}
  ~AscendDeviceContext() override = default;

  // Initialize the device context.
  void Initialize() override;

  // Destroy device context and release device resource.
  void Destroy() override;

  bool PartitionGraph(const FuncGraphPtr &func_graph) const override;

  RunMode GetRunMode(const FuncGraphPtr &func_graph) const override;

  DeprecatedInterface *GetDeprecatedInterface() override;

 private:
  bool initialized_{false};
  AscendKernelRuntime *runtime_instance_{nullptr};
  std::unique_ptr<AscendDeprecatedInterface> deprecated_interface_;
};

// Some NOP nodes have be hide in execution order, it doesn't have output device address, this function creates
// output device address for these nodes, and the output device address is the same with input device address.
void AssignOutputNopNodeDeviceAddress(const KernelGraphPtr &graph, const device::DeviceContext *device_context);
}  // namespace ascend
}  // namespace device
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_HARDWARE_ASCEND_ASCEND_DEVICE_CONTEXT_H_
