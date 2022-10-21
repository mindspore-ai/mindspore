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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_APPLY_ADAGRAD_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_APPLY_ADAGRAD_CPU_KERNEL_H_

#include <thread>
#include <vector>
#include <map>
#include <utility>
#include <string>

#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"

namespace mindspore {
namespace kernel {
class ApplyAdagradCpuKernelMod : public NativeCpuKernelMod {
 public:
  ApplyAdagradCpuKernelMod() = default;
  ~ApplyAdagradCpuKernelMod() override = default;

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override {
    return kernel_func_(this, inputs, workspace, outputs);
  }

  std::vector<KernelAttr> GetOpSupport() override;

 private:
  template <typename T>
  bool LaunchKernel(const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &workspace,
                    const std::vector<kernel::AddressPtr> &outputs);

  using ApplyAdagradFunc =
    std::function<bool(ApplyAdagradCpuKernelMod *, const std::vector<kernel::AddressPtr> &,
                       const std::vector<kernel::AddressPtr> &, const std::vector<kernel::AddressPtr> &)>;
  static std::vector<std::pair<KernelAttr, ApplyAdagradFunc>> func_list_;
  ApplyAdagradFunc kernel_func_;

  void CheckParam(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs) const;

  template <typename T>
  void LaunchApplyAdagrad(T *var, T *accum, const T *lr, const T *gradient, size_t start, size_t end) const;

  bool update_slots_{true};
  TypeId dtype_{kTypeUnknown};
  ShapeVector input_shape_;
};
}  // namespace kernel
}  // namespace mindspore
#endif
