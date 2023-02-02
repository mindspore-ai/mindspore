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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_SCALAR_BITWISE_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_SCALAR_BITWISE_CPU_KERNEL_H_
#include <vector>
#include <memory>
#include <utility>
#include <map>
#include <string>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"

namespace mindspore {
namespace kernel {
class ScalarBitwiseCpuKernelMod : public NativeCpuKernelMod {
 public:
  ScalarBitwiseCpuKernelMod() = default;
  explicit ScalarBitwiseCpuKernelMod(const std::string &kernel_type) : kernel_type_(kernel_type) {}
  ~ScalarBitwiseCpuKernelMod() override = default;

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  int Resize(
    const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
    const std::vector<KernelTensorPtr> &outputs,
    const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost = std::map<uint32_t, tensor::TensorPtr>()) override;

  bool Launch(const std::vector<KernelTensorPtr> &inputs, const std::vector<KernelTensorPtr> &outputs,
              const std::vector<AddressPtr> &workspace) override {
    return kernel_func_(this, inputs, outputs, workspace);
  }

  std::vector<KernelAttr> GetOpSupport() override;

 private:
  template <typename T, typename S, typename N>
  bool LaunchKernel(const std::vector<KernelTensorPtr> &inputs, const std::vector<KernelTensorPtr> &outputs,
                    const std::vector<AddressPtr> &);

  using ScalarBitwiseFunc =
    std::function<bool(ScalarBitwiseCpuKernelMod *, const std::vector<kernel::KernelTensorPtr> &,
                       const std::vector<kernel::KernelTensorPtr> &, const std::vector<kernel::AddressPtr> &)>;
  static std::vector<std::pair<KernelAttr, ScalarBitwiseFunc>> func_list_;
  ScalarBitwiseFunc kernel_func_;
  std::string kernel_type_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_SCALAR_BITWISE_CPU_KERNEL_H_
