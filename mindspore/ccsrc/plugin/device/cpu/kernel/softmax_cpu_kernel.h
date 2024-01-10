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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_SOFTMAX_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_SOFTMAX_CPU_KERNEL_H_

#include <functional>
#include <vector>
#include <utility>
#include <map>
#include "plugin/device/cpu/kernel/cpu_kernel.h"

namespace mindspore {
namespace kernel {
class SoftmaxCpuKernelMod final : public NativeCpuKernelMod {
 public:
  SoftmaxCpuKernelMod() = default;
  ~SoftmaxCpuKernelMod() override = default;

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs) override {
    return kernel_func_(this, inputs, workspace, outputs);
  }

  std::vector<KernelAttr> GetOpSupport() override;

 private:
  using LaunchFunc =
    std::function<bool(SoftmaxCpuKernelMod *, const std::vector<KernelTensor *> &inputs,
                       const std::vector<KernelTensor *> &workspace, const std::vector<KernelTensor *> &outputs)>;
  static std::vector<std::pair<KernelAttr, LaunchFunc>> func_list_;
  LaunchFunc kernel_func_;

  template <typename T>
  bool LaunchKernel(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                    const std::vector<KernelTensor *> &outputs) noexcept;

  void CheckAndRectifyAxis(KernelTensor *axis_kernel_tensor) noexcept;

  int64_t axis_{0};
  int64_t input_dims_{0};
  int64_t dim_axis_{0};
  bool last_axis_{false};
  std::vector<int64_t> input_shape_;
  size_t inner_size_{0};
  size_t input_elements_{0};
  size_t output_elements_{0};
  size_t unit_size_{0};
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_SOFTMAX_CPU_KERNEL_H_
