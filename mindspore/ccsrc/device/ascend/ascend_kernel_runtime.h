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
#ifndef MINDSPORE_CCSRC_DEVICE_ASCEND_ASCEND_KERNEL_RUNTIME_H_
#define MINDSPORE_CCSRC_DEVICE_ASCEND_ASCEND_KERNEL_RUNTIME_H_
#include <memory>
#include <vector>
#include <string>
#include <unordered_map>
#include "device/kernel_runtime.h"
#include "runtime/context.h"
#include "framework/ge_runtime/davinci_model.h"
#include "device/kernel_runtime_manager.h"
#include "session/session_basic.h"

using ge::model_runner::TaskInfo;
using std::unordered_map;
using std::vector;
namespace mindspore {
namespace device {
namespace ascend {
class AscendKernelRuntime : public KernelRuntime {
 public:
  AscendKernelRuntime() = default;
  ~AscendKernelRuntime() override;
  bool Init() override;
  bool DumpData(session::KernelGraph *graph) override;
  bool GenTask(const session::KernelGraph *graph) override;
  bool RunTask(const session::KernelGraph *graph) override;
  bool LoadTask(const session::KernelGraph *graph) override;

 protected:
  DeviceAddressPtr CreateDeviceAddress(void *device_ptr, size_t device_size, const string &format,
                                       TypeId type_id) override;
  bool SyncStream() override;

 private:
  bool InitDevice();
  bool ResetDevice();
  bool HcclInit();
  bool NeedDestroyHccl();
  bool DestroyHccl();

  void ClearGraphModelMap();
  void ReleaseDeviceRes() override;
  bool GraphWithEmptyTaskList(const session::KernelGraph *graph) const;
  bool CheckGraphIdValid(GraphId graph_id) const;
  static void DebugTaskIdName(GraphId graph_id) ;

  rtContext_t rt_context_{nullptr};
  bool initialized_{false};
  unordered_map<GraphId, vector<std::shared_ptr<TaskInfo>>> task_map_;
  unordered_map<GraphId, std::shared_ptr<ge::model_runner::DavinciModel>> graph_model_map_;
};

MS_REG_KERNEL_RUNTIME(kAscendDevice, AscendKernelRuntime);
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_DEVICE_ASCEND_ASCEND_KERNEL_RUNTIME_H_
