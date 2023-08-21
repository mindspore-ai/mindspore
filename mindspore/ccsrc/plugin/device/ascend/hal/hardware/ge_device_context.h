/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_RUNTIME_HARDWARE_ASCEND_GE_DEVICE_CONTEXT_H_
#define MINDSPORE_CCSRC_RUNTIME_HARDWARE_ASCEND_GE_DEVICE_CONTEXT_H_

#include <memory>
#include <string>
#include <map>
#include "plugin/device/ascend/hal/hardware/ascend_deprecated_interface.h"
#include "runtime/hardware/device_context.h"
#include "runtime/device/memory_manager.h"
#include "utils/ms_context.h"
#include "include/transform/graph_ir/types.h"
#include "plugin/device/ascend/hal/hardware/ascend_collective_comm_lib.h"
#include "plugin/device/ascend/hal/hardware/ge_kernel_executor.h"
#include "plugin/device/ascend/hal/hardware/ge_graph_executor.h"
#include "plugin/device/ascend/hal/hardware/ge_device_res_manager.h"

namespace mindspore {
namespace device {
namespace ascend {
class GeGraphExecutor;
class GeKernelExecutor;
class GeDeviceResManager;

class GeDeviceContext : public DeviceInterface<GeGraphExecutor, GeKernelExecutor, GeDeviceResManager> {
 public:
  explicit GeDeviceContext(const DeviceContextKey &device_context_key)
      : DeviceInterface(device_context_key), initialized_(false) {}
  ~GeDeviceContext() override = default;

  void Initialize() override;

  void Destroy() override;

  bool PartitionGraph(const FuncGraphPtr &func_graph) const override;
  RunMode GetRunMode(const FuncGraphPtr &func_graph) const override;

  DeprecatedInterface *GetDeprecatedInterface() override;

 private:
  DISABLE_COPY_AND_ASSIGN(GeDeviceContext);

  void InitGe(const std::shared_ptr<MsContext> &inst_context);
  bool FinalizeGe(const std::shared_ptr<MsContext> &inst_context);
  void GetGeOptions(const std::shared_ptr<MsContext> &inst_context, std::map<std::string, std::string> *ge_options);
  void SetHcclOptions(const std::shared_ptr<MsContext> &inst_context, std::map<std::string, std::string> *ge_options);
  void SetAscendConfig(const std::shared_ptr<MsContext> &ms_context_ptr,
                       std::map<std::string, std::string> *ge_options) const;
  void SetDisableReuseMemoryFlag(std::map<std::string, std::string> *ge_options) const;
  void SetDumpOptions(std::map<std::string, std::string> *ge_options) const;
  void InitDump() const;
  void FinalizeDump() const;

  std::unique_ptr<AscendDeprecatedInterface> deprecated_interface_;
  bool initialized_;
};
}  // namespace ascend
}  // namespace device
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_HARDWARE_ASCEND_GE_DEVICE_CONTEXT_H_
