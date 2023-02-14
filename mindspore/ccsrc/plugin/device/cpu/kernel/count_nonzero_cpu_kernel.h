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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_COUNT_NONZERO_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_COUNT_NONZERO_CPU_KERNEL_H_

#include <memory>
#include <vector>
#include <iostream>
#include <string>
#include <map>
#include <utility>

#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"
#include "plugin/device/cpu/kernel/nnacl/arithmetic.h"

namespace mindspore {
namespace kernel {
class CountNonZeroCpuKernelMod : public NativeCpuKernelMod {
 public:
  CountNonZeroCpuKernelMod() = default;
  ~CountNonZeroCpuKernelMod() override = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override {
    return kernel_func_(this, inputs, outputs);
  }

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs, const std::map<uint32_t, tensor::TensorPtr> &) override;

 protected:
  std::vector<KernelAttr> GetOpSupport() override;

 private:
  template <typename T>
  bool LaunchKernel(const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &outputs);
  void ComputeCountParameter(void);
  using CountNonZeroLaunchFunc = std::function<bool(CountNonZeroCpuKernelMod *, const std::vector<kernel::AddressPtr> &,
                                                    const std::vector<kernel::AddressPtr> &)>;

  static std::vector<std::pair<KernelAttr, CountNonZeroLaunchFunc>> func_list_;
  CountNonZeroLaunchFunc kernel_func_;
  std::vector<int64_t> dims_;
  float value_;
  ShapeVector x_shape_;
  ShapeVector y_shape_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_COUNT_NONZERO_CPU_KERNEL_H_
