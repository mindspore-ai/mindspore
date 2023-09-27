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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_APPLY_ADAGRAD_V2_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_APPLY_ADAGRAD_V2_CPU_KERNEL_H_

#include <map>
#include <vector>
#include <utility>

#include "mindspore/core/ops/apply_adagrad_v2.h"
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"

namespace mindspore {
namespace kernel {
class ApplyAdagradV2CpuKernelMod : public NativeCpuKernelMod {
 public:
  ApplyAdagradV2CpuKernelMod() = default;
  ~ApplyAdagradV2CpuKernelMod() override = default;

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &,
              const std::vector<KernelTensor *> &outputs) override {
    return kernel_func_(this, inputs, outputs);
  }

  std::vector<KernelAttr> GetOpSupport() override;

 private:
  int CheckParam(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) const;

  template <typename T>
  bool LaunchKernel(const std::vector<kernel::KernelTensor *> &inputs,
                    const std::vector<kernel::KernelTensor *> &outputs);
  using ApplyAdagradV2Func =
    std::function<bool(ApplyAdagradV2CpuKernelMod *, const std::vector<kernel::KernelTensor *> &,
                       const std::vector<kernel::KernelTensor *> &)>;
  static std::vector<std::pair<KernelAttr, ApplyAdagradV2Func>> func_list_;
  ApplyAdagradV2Func kernel_func_;

  float epsilon_;
  bool update_slots_{true};
  int64_t batch_rank_{0};
  size_t inner_input_size_;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_APPLY_ADAGRAD_V2_CPU_KERNEL_H_
