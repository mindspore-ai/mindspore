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
#include <map>

#include "plugin/device/cpu/kernel/sparse_optimizer_cpu_kernel.h"
#include "plugin/factory/ms_factory.h"

namespace mindspore {
namespace kernel {
class SparseApplyCenteredRMSPropCpuKernelMod : public SparseOptimizerCpuKernelMod,
                                               public MatchKernelHelper<SparseApplyCenteredRMSPropCpuKernelMod> {
 public:
  SparseApplyCenteredRMSPropCpuKernelMod() = default;
  ~SparseApplyCenteredRMSPropCpuKernelMod() override = default;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs) override {
    return kernel_func_(this, inputs, workspace, outputs);
  }

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  const std::vector<std::pair<KernelAttr, KernelRunFunc>> &GetFuncList() const override;

  template <typename I, typename T>
  bool LaunchKernel(const std::vector<KernelTensor *> &inputs, const std::vector<kernel::KernelTensor *> &workspace,
                    const std::vector<KernelTensor *> &outputs);

  std::vector<KernelAttr> GetOpSupport() override { return OpSupport(); }

 protected:
  void ResetResource() noexcept;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_SPARSE_APPLY_CENTERED_RMS_PROP_CPU_KERNEL_H_
