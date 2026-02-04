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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_ARRAYS_DIAG_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_ARRAYS_DIAG_GPU_KERNEL_H_

#include <vector>
#include <memory>
#include <utility>
#include <map>
#include "kernel/gpu/gpu_kernel.h"
#include "include/runtime/hardware_abstract/kernel_base/ms_factory.h"

namespace mindspore {
namespace kernel {
class DiagGpuKernelMod : public NativeGpuKernelMod {
 public:
  DiagGpuKernelMod() = default;
  ~DiagGpuKernelMod() override = default;

  std::vector<KernelAttr> GetOpSupport() override;

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override {
    return kernel_launch_func_(this, inputs, workspace, outputs, stream_ptr);
  }

 private:
  template <typename DataType>
  bool LaunchKernel(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                    const std::vector<KernelTensor *> &outputs, void *stream_ptr);

  using DiagLaunchFunc =
    std::function<bool(DiagGpuKernelMod *, const std::vector<KernelTensor *> &, const std::vector<KernelTensor *> &,
                       const std::vector<KernelTensor *> &, void *)>;
  static std::vector<std::pair<KernelAttr, DiagLaunchFunc>> diag_func_list_;
  DiagLaunchFunc kernel_launch_func_;

  // Support the batch calculation of vmap.
  int64_t batch_rank_{0};
  size_t batch_size_{0};

  size_t input_size_{0};
  size_t output_size_{0};
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_ARRAYS_DIAG_GPU_KERNEL_H_
