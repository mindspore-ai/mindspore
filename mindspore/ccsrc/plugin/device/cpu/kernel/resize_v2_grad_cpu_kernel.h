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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_RESIZE_V2_GRAD_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_RESIZE_V2_GRAD_CPU_KERNEL_H_

#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <unordered_map>
#include <vector>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"

namespace mindspore {
namespace kernel {
constexpr auto kUnknown = "Unknown";

class ResizeV2GradCpuKernelMod : public NativeCpuKernelMod, public MatchKernelHelper<ResizeV2GradCpuKernelMod> {
 public:
  ResizeV2GradCpuKernelMod() = default;
  ~ResizeV2GradCpuKernelMod() override = default;

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override {
    return kernel_func_(this, inputs, workspace, outputs);
  }

  int Resize(
    const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
    const std::vector<KernelTensorPtr> &outputs,
    const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost = std::map<uint32_t, tensor::TensorPtr>()) override;

  const std::vector<std::pair<KernelAttr, KernelRunFunc>> &GetFuncList() const override;

 protected:
  std::vector<KernelAttr> GetOpSupport() override { return OpSupport(); }

 private:
  template <typename T>
  bool LaunchKernel(const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &workspace,
                    const std::vector<kernel::AddressPtr> &outputs);

  template <typename T>
  bool LaunchKernelByCubic(const std::vector<kernel::AddressPtr> &inputs,
                           const std::vector<kernel::AddressPtr> &outputs);

  template <typename T>
  bool LaunchKernelByLinear(const std::vector<kernel::AddressPtr> &inputs,
                            const std::vector<kernel::AddressPtr> &outputs);

  template <typename T>
  bool LaunchKernelByNearest(const std::vector<kernel::AddressPtr> &inputs,
                             const std::vector<kernel::AddressPtr> &outputs);

  std::string kernel_type_{kUnknown};
  std::string kernel_name_;
  bool align_corners_{false};
  std::string mode_{"nearest"};
  TypeId sizes_dtype_{kTypeUnknown};
  float width_scale_;
  float height_scale_;
  size_t batch_{0};
  size_t channel_{0};
  size_t channels_{0};
  size_t in_width_{0};
  size_t in_height_{0};
  size_t out_width_{0};
  size_t out_height_{0};
  size_t in_hw_size_{0};
  size_t out_hw_size_{0};
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_RESIZE_V2_GRAD_CPU_KERNEL_H_
