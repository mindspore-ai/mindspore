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
#include <string>
#include <utility>
#include <mutex>
#include "include/runtime/hardware_abstract/device_context/device_context.h"
#include "include/runtime/hardware_abstract/device_context/device_context_manager.h"
#include "include/runtime/hardware_abstract/collective/collective_communication_lib.h"
#include "include/runtime/hardware_abstract/collective/collective_comm_lib_loader.h"
#include "plugin/cpu/res_manager/cpu_res_manager.h"

namespace mindspore {
namespace device {
namespace cpu {
class CPUKernelExecutor : public KernelExecutor {
 public:
  CPUKernelExecutor() = default;
  ~CPUKernelExecutor() override = default;

  void OptimizeGraph(const FuncGraphPtr &graph) const override;

  void CreateKernel(const std::vector<CNodePtr> &nodes) const override;
  kernel::KernelModPtr CreateKernelMod(const std::string &op_name) const override;

  // Kernel that is not supported by other device can be backed off and rebuilt on the CPU.
  // The function will set kernel info and create kernel mod.
  void RebuildKernelSelectBackoffOp(const std::vector<CNodePtr> &nodes) const override;

  void PreprocessBeforeRun(const FuncGraphPtr &graph) const override;

  bool LaunchKernel(const CNodePtr &kernel, const std::vector<KernelTensor *> &inputs,
                    const std::vector<KernelTensor *> &workspace, const std::vector<KernelTensor *> &outputs,
                    KernelMod *kernel_mod, void * /* stream */) const override;
  bool LaunchKernelHP(const CNodePtr &kernel, const std::vector<KernelTensor *> &inputs,
                      const std::vector<KernelTensor *> &workspace, const std::vector<KernelTensor *> &outputs,
                      KernelMod *kernel_mod, void *stream) const override {
    return LaunchKernel(kernel, inputs, workspace, outputs, kernel_mod, stream);
  }

  bool ExecuteKernelTask(const runtime::KernelTaskType &task_type, const tensor::TensorPtrList &input_tensors,
                         const tensor::TensorPtrList &output_tensors, const size_t &stream_id) const override;
  bool ExecuteKernelTask(const runtime::KernelTaskType &task_type,
                         const std::vector<KernelTensor *> &input_kernel_tensors,
                         const std::vector<KernelTensor *> &output_kernel_tensors,
                         const size_t &stream_id) const override;

  std::vector<size_t> GetLaunchIgnoredInputAddressIdx(const AnfNodePtr &node) const override;

  bool IsLaunchIgnoredInputAddressIdx(const AnfNodePtr &node, size_t input_idx) const override;

 private:
  // Select the matching backend kernels according to the data type and format of input and output for all
  // execution operators, and set final device data type and format information for backend kernels, device
  // data type and format which replace original data type and format will use for executing kernels.
  void SetOperatorInfo(const KernelGraphPtr &graph) const;
  void SingleOpGraphOptimize(const KernelGraphPtr &graph) const;
  void OptimizeGraphImpl(const KernelGraphPtr &graph) const;
  void OptimizeMindIR(const KernelGraphPtr &graph) const;
  // Launch a kernel and record the elapsed time end to end.
  bool LaunchKernelWithProfiling(const CNodePtr &kernel, const std::vector<KernelTensor *> &inputs,
                                 const std::vector<KernelTensor *> &workspace,
                                 const std::vector<KernelTensor *> &outputs, KernelMod *kernel_mod) const;
  // Launch a kernel by 'KernelMod' of the kernel.
  bool DoLaunchKernel(const CNodePtr &kernel, const std::vector<KernelTensor *> &inputs,
                      const std::vector<KernelTensor *> &workspace, const std::vector<KernelTensor *> &outputs,
                      KernelMod *kernel_mod) const;
  void UpdateKernelRefInfo(const KernelGraphPtr &graph) const;

  mutable std::mutex launch_mutex_;
};

class CPUDeviceContext : public DeviceInterface<CPUKernelExecutor, CPUResManager> {
 public:
  explicit CPUDeviceContext(const DeviceContextKey &device_context_key) : DeviceInterface(device_context_key) {}
  ~CPUDeviceContext() override = default;

  void Initialize() override;

  void Destroy() override;

 private:
  DISABLE_COPY_AND_ASSIGN(CPUDeviceContext);
};
}  // namespace cpu
}  // namespace device
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_HARDWARE_CPU_CPU_DEVICE_CONTEXT_H_
