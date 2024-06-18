/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_RUNTIME_HARDWARE_ASCEND_GE_KERNEL_EXECUTOR_H_
#define MINDSPORE_CCSRC_RUNTIME_HARDWARE_ASCEND_GE_KERNEL_EXECUTOR_H_

#include <vector>
#include <memory>
#include <string>
#include <set>
#include <map>
#include <utility>
#include "runtime/hardware/device_context.h"
#include "runtime/device/memory_manager.h"
#include "runtime/device/kernel_runtime.h"
#include "plugin/device/ascend/hal/device/ascend_device_address.h"
#include "plugin/device/ascend/hal/hardware/ge_graph_executor.h"
#include "plugin/device/ascend/mindio/mindio_adapter.h"

namespace mindspore {
namespace device {
namespace ascend {
class GeKernelExecutor : public KernelExecutor {
 public:
  GeKernelExecutor() = default;
  ~GeKernelExecutor() override = default;

  void Initialize() override;
  void Destroy() override;

  // Optimize the kernel graph for graph mode.
  void OptimizeGraph(const FuncGraphPtr &graph) const override;

  // Generate 'KernelMod' for all kernels and set 'KernelMod' into kernel,
  // 'KernelMod' is real executive object of kernel.
  void CreateKernel(const std::vector<CNodePtr> &nodes) const override;

  // Generate 'KernelMod' by operator name.
  // Note: Only support generage aclnn kernel mod current.
  kernel::KernelModPtr CreateKernelMod(const std::string &op_name) const override;

  // Adjust kernel graph before run graph, used in Graph Mode.
  void PreprocessBeforeRun(const FuncGraphPtr &graph) const override;

  // Launch a kernel via 'KernelMod' of the kernel.
  bool LaunchKernel(const CNodePtr &kernel, const std::vector<KernelTensor *> &inputs,
                    const std::vector<KernelTensor *> &workspace, const std::vector<KernelTensor *> &outputs,
                    KernelMod *kernel_mod, void *stream) const override;
  bool LaunchCallback(CallbackFunc callback_func, size_t stream_id) const;

  // Unify the MindIR, the default behavior uses the common unified MindIR.
  void UnifyMindIR(const KernelGraphPtr &graph) const override;
  void AddMindIRPass(const KernelGraphPtr &graph) const override;
  void OptimizeExecutionOrder(const FuncGraphPtr &graph) const;

  // Get rank id for distributed training.
  uint32_t GetRankID() const override { return 0; }

  bool ExecuteKernelTask(const runtime::KernelTaskType &task_type, const device::DeviceAddressPtrList &input_addr_list,
                         const device::DeviceAddressPtrList &output_addr_list, const size_t &stream_id) const override;

 private:
  static void DoSomas(const FuncGraphPtr &graph, const std::vector<std::pair<CNodePtr, CNodePtr>> &sched_events);
  static void DoStreamAssign(const KernelGraphPtr &kernel_graph,
                             const std::vector<std::pair<CNodePtr, CNodePtr>> &sched_events);
  // launch
  bool MemoryCopyAsync(const CNodePtr &node, const vector<KernelTensor *> &inputs,
                       const vector<KernelTensor *> &outputs) const;
  bool PySyncRuning(void *stream) const;
  void DoAsyncCkpt(const CNodePtr &kernel) const;

  mutable std::set<CNodePtr> nop_op_to_memcpy_;
  // Maybe AscendDeviceResManager and GEDeviceResManager now
  DeviceResManager *res_manager_{nullptr};
  GeGraphExecutor *graph_executor_{nullptr};
  bool initialized_ = false;
};
}  // namespace ascend
}  // namespace device
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_HARDWARE_ASCEND_GE_KERNEL_EXECUTOR_H_
