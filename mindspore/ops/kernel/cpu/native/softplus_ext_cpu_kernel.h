/**
 * Copyright 2026 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_SOFTPLUS_EXT_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_SOFTPLUS_EXT_CPU_KERNEL_H_

#include <utility>
#include <vector>
#include "include/runtime/hardware_abstract/kernel_base/common_utils.h"
#include "include/runtime/hardware_abstract/kernel_base/ms_factory.h"
#include "kernel/cpu/cpu_kernel.h"

namespace mindspore {
namespace kernel {
namespace softplus_ext_cpu {
class SoftplusExtCpuKernelMod : public NativeCpuKernelMod, public MatchKernelHelper<SoftplusExtCpuKernelMod> {
 public:
  SoftplusExtCpuKernelMod() = default;
  ~SoftplusExtCpuKernelMod() override = default;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs) override {
    return kernel_func_(this, inputs, workspace, outputs);
  }

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;
  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;
  const std::vector<std::pair<KernelAttr, KernelRunFunc>> &GetFuncList() const override;
  std::vector<KernelAttr> GetOpSupport() override { return OpSupport(); }

 private:
  template <typename T, typename TBeta, typename TThreshold>
  bool LaunchKernel(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                    const std::vector<KernelTensor *> &outputs);

  size_t size_{0};
};
}  // namespace softplus_ext_cpu
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_SOFTPLUS_EXT_CPU_KERNEL_H_
