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
#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_APPLY_GRADIENT_DESCENT_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_APPLY_GRADIENT_DESCENT_CPU_KERNEL_H_
#include <algorithm>
#include <map>
#include <vector>
#include <memory>
#include <string>
#include <utility>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"

namespace mindspore {
namespace kernel {
class ApplyGradientDescentCpuKernelMod : public NativeCpuKernelMod {
 public:
  ApplyGradientDescentCpuKernelMod() = default;
  ~ApplyGradientDescentCpuKernelMod() override = default;

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs) override {
    return kernel_func_(this, inputs, outputs);
  }

  template <typename T>
  bool LaunchKernel(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs);

  std::vector<KernelAttr> GetOpSupport() override;

 private:
  size_t input_size_;
  size_t inner_input_size_;  // inner_input_size_ is the number of elements in one batch.
  int64_t batch_rank_{0};
  TypeId dtype_{kTypeUnknown};
  template <typename T>
  void ComputeTask(T *x_data_addr, T *grid_data_addr, T *output_data_addr, const size_t &seq);

  using ApplyGradientDescentLaunchFunc = std::function<bool(
    ApplyGradientDescentCpuKernelMod *, const std::vector<KernelTensor *> &, const std::vector<KernelTensor *> &)>;
  static std::vector<std::pair<KernelAttr, ApplyGradientDescentLaunchFunc>> func_list_;
  ApplyGradientDescentLaunchFunc kernel_func_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_APPLY_GRADIENT_DESCENT_CPU_KERNEL_H_
