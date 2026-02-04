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

#ifndef MINDSPORE_CCSRC_RUNTIME_HARDWARE_GPU_GPU_DEVICE_CONTEXT_H_
#define MINDSPORE_CCSRC_RUNTIME_HARDWARE_GPU_GPU_DEVICE_CONTEXT_H_

#include <tuple>
#include <vector>
#include <memory>
#include <string>
#include <utility>
#include <unordered_map>
#include "include/runtime/hardware_abstract/device_context/device_context.h"
#include "include/runtime/hardware_abstract/device_context/device_context_manager.h"
#include "kernel/gpu/cuda_impl/cuda_ops/cuda_device_info.h"
#include "plugin/gpu/res_manager/gpu_res_manager.h"

namespace mindspore {
namespace device {
namespace gpu {
class GPUKernelExecutor : public KernelExecutor {
 public:
  GPUKernelExecutor() = default;
  ~GPUKernelExecutor() override = default;

  void Initialize() override;
  void Destroy() override;

  // Optimize the kernel graph for graph mode.
  void OptimizeGraph(const FuncGraphPtr &graph) const override;

  void CreateKernel(const std::vector<CNodePtr> &nodes) const override;
  kernel::KernelModPtr CreateKernelMod(const std::string &op_name) const override;

  void PreprocessBeforeRun(const FuncGraphPtr &graph) const override;

  bool LaunchKernel(const CNodePtr &kernel, const std::vector<KernelTensor *> &inputs,
                    const std::vector<KernelTensor *> &workspace, const std::vector<KernelTensor *> &outputs,
                    KernelMod *kernel_mod, void *stream) const override;
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

  // General graph optimezer ignore device data type and format.
  void OptimizeGraphWithoutDeviceInfo(const KernelGraphPtr &graph) const;
  // Optimize the kernel graph according to device type, such format transform.
  void OptimizeGraphWithDeviceInfo(const KernelGraphPtr &graph) const;

  // Operator fusion optimization.
  void FuseOperators(const KernelGraphPtr &graph) const;

  // Update kernel ref info before create kernel
  void UpdateKernelRefInfo(const KernelGraphPtr &graph) const;

  // Launch a kernel and record the elapsed time end to end.
  bool LaunchKernelWithProfiling(const CNodePtr &kernel, const std::vector<KernelTensor *> &inputs,
                                 const std::vector<KernelTensor *> &workspace,
                                 const std::vector<KernelTensor *> &outputs, KernelMod *kernel_mod, void *stream) const;
  // Launch a kernel by 'KernelMod' of the kernel.
  bool DoLaunchKernel(const CNodePtr &kernel, const std::vector<KernelTensor *> &inputs,
                      const std::vector<KernelTensor *> &workspace, const std::vector<KernelTensor *> &outputs,
                      KernelMod *kernel_mod, void *stream) const;

  // The cublas handle is not thread safety specifically, it is not recommended that multiple threads access the same
  // cublas handle at the same time, so need the launch mutex when multiple threads launch the cublas kernels.
  mutable std::mutex launch_mutex_;
  GPUResManager *res_manager_{nullptr};
  bool initialized_ = false;
};

class GPUDeviceContext : public DeviceInterface<GPUKernelExecutor, GPUResManager> {
 public:
  explicit GPUDeviceContext(const DeviceContextKey &device_context_key) : DeviceInterface(device_context_key) {}
  ~GPUDeviceContext() override = default;

  // Set device id and initialize device resource, such as stream, cudnn and cublas handle.
  void Initialize() override;

  // Release device memory, stream, cudnn and cublas handle, etc.
  void Destroy() override;

  static uint32_t GetDeviceCount();
  static std::string GetDeviceName(uint32_t device_id);
  static std::tuple<int, int> GetDeviceCapability(uint32_t device_id);
  static cudaDeviceProp GetDeviceProperties(uint32_t device_id);
  static std::string GetArchList();

 private:
  DISABLE_COPY_AND_ASSIGN(GPUDeviceContext);
};
}  // namespace gpu
}  // namespace device
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_RUNTIME_HARDWARE_GPU_GPU_DEVICE_CONTEXT_H_
