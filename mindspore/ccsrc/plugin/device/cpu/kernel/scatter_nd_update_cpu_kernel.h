/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_SCATTER_ND_UPDATE_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_SCATTER_ND_UPDATE_CPU_KERNEL_H_

#include <vector>
#include <memory>
#include <algorithm>
#include <map>
#include <utility>
#include <string>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"
#include "kernel/common_utils.h"

namespace mindspore {
namespace kernel {
class ScatterUpdateArithmeticCpuKernelMod : public NativeCpuKernelMod {
 public:
  ScatterUpdateArithmeticCpuKernelMod() = default;
  explicit ScatterUpdateArithmeticCpuKernelMod(const std::string &kernel_name) : kernel_type_(kernel_name) {}
  ~ScatterUpdateArithmeticCpuKernelMod() override = default;

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs, const std::map<uint32_t, tensor::TensorPtr> &) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override {
    return kernel_func_(this, inputs, workspace, outputs);
  }

  std::vector<KernelAttr> GetOpSupport() override;

 private:
  template <typename T, typename S>
  bool LaunchKernel(const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &workspace,
                    const std::vector<kernel::AddressPtr> &outputs);
  using LaunchFunc =
    std::function<bool(ScatterUpdateArithmeticCpuKernelMod *, const std::vector<kernel::AddressPtr> &,
                       const std::vector<kernel::AddressPtr> &, const std::vector<kernel::AddressPtr> &)>;
  static std::map<std::string, std::vector<std::pair<KernelAttr, LaunchFunc>>> kernel_attr_map_;

  LaunchFunc kernel_func_{};
  TypeId dtype_value_{kTypeUnknown};
  TypeId dtype_shape_{kTypeUnknown};
  int unit_size_{0};
  size_t num_units_{0};
  size_t indices_unit_rank_{0};
  std::string kernel_type_;
  std::vector<size_t> out_strides_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_SCATTER_ND_UPDATE_CPU_KERNEL_H_
