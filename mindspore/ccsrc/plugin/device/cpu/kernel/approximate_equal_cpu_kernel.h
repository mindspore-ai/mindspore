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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_APPROXIMATE_EQUAL_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_APPROXIMATE_EQUAL_CPU_KERNEL_H_

#include <vector>
#include <utility>
#include "mindspore/core/ops/approximate_equal.h"
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"

namespace mindspore {
namespace kernel {
class ApproximateEqualCpuKernelMod : public NativeCpuKernelMod {
 public:
  ApproximateEqualCpuKernelMod() {}
  ~ApproximateEqualCpuKernelMod() override = default;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs) override {
    return kernel_func_(this, inputs, outputs);
  }

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  std::vector<KernelAttr> GetOpSupport() override;

 private:
  template <typename T>
  bool LaunchKernel(const std::vector<kernel::KernelTensor *> &inputs,
                    const std::vector<kernel::KernelTensor *> &outputs);

  using ApproximateEqualFunc =
    std::function<bool(ApproximateEqualCpuKernelMod *, const std::vector<kernel::KernelTensor *> &,
                       const std::vector<kernel::KernelTensor *> &)>;

  void CheckParam(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) const;

  size_t output_size_{1};
  float tolerance_{1e-5};
  ApproximateEqualFunc kernel_func_;
  std::vector<size_t> input_shape_;
  std::vector<size_t> output_shape_;
  static std::vector<std::pair<KernelAttr, ApproximateEqualFunc>> func_list_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_APPROXIMATE_EQUAL_CPU_KERNEL_H_
