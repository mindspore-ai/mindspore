/**
 * Copyright 2022 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * htp://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_SPARSE_APPLY_CENTERED_RMS_PROP_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_SPARSE_APPLY_CENTERED_RMS_PROP_CPU_KERNEL_H_

#include <utility>
#include <vector>

#include "plugin/device/cpu/kernel/sparse_optimizer_cpu_kernel.h"
#include "plugin/factory/ms_factory.h"

namespace mindspore {
namespace kernel {
class SparseApplyCenteredRMSPropCpuKernelMod : public SparseOptimizerCpuKernelMod {
 public:
  SparseApplyCenteredRMSPropCpuKernelMod() = default;
  ~SparseApplyCenteredRMSPropCpuKernelMod() override = default;

  void InitKernel(const CNodePtr &kernel_node) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override {
    return kernel_func_(this, inputs, outputs);
  };

  template <typename I, typename T>
  bool LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs);

 protected:
  std::vector<KernelAttr> GetOpSupport() override;

 private:
  using SparseApplyCenteredRMSPropFunc =
    std::function<bool(SparseApplyCenteredRMSPropCpuKernelMod *, const std::vector<kernel::AddressPtr> &,
                       const std::vector<kernel::AddressPtr> &)>;
  static std::vector<std::pair<KernelAttr, SparseApplyCenteredRMSPropFunc>> func_list_;
  SparseApplyCenteredRMSPropFunc kernel_func_;
  size_t indices_size_{0};
  size_t var_first_dim_size_{0};
  size_t var_outer_dim_size_{1};
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_SPARSE_APPLY_CENTERED_RMS_PROP_CPU_KERNEL_H_
