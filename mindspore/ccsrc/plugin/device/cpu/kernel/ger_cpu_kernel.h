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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_GER_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_GER_CPU_KERNEL_H_

#include <functional>
#include <memory>
#include <vector>
#include <iostream>
#include <string>
#include <complex>
#include <map>
#include <utility>

#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"
#include "plugin/device/cpu/kernel/nnacl/arithmetic_parameter.h"

namespace mindspore {
namespace kernel {
class GerCpuKernelMod : public NativeCpuKernelMod, public MatchKernelHelper<GerCpuKernelMod> {
 public:
  GerCpuKernelMod() = default;
  explicit GerCpuKernelMod(const std::string &kernel_type) : kernel_type_(kernel_type) {}
  ~GerCpuKernelMod() override = default;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs) override {
    return kernel_func_(this, inputs, workspace, outputs);
  }

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  const std::vector<std::pair<KernelAttr, KernelRunFunc>> &GetFuncList() const override;

  std::vector<KernelAttr> GetOpSupport() override { return OpSupport(); }

 private:
  template <typename T>
  bool LaunchKernel(const std::vector<kernel::KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                    const std::vector<kernel::KernelTensor *> &outputs);
  using LaunchFunc = std::function<bool(GerCpuKernelMod *, const std::vector<kernel::KernelTensor *> &inputs,
                                        const std::vector<kernel::KernelTensor *> &workspace,
                                        const std::vector<kernel::KernelTensor *> &outputs)>;
  LaunchFunc launch_func_;

  template <typename T>
  void InitLaunchFunc();

  template <typename T>
  bool LaunchBatchesElse(const std::vector<kernel::KernelTensor *> &inputs,
                         const std::vector<kernel::KernelTensor *> &workspace,
                         const std::vector<kernel::KernelTensor *> &outputs);
  template <typename T>
  bool LaunchNoBatchesElse(const std::vector<kernel::KernelTensor *> &inputs,
                           const std::vector<kernel::KernelTensor *> &workspace,
                           const std::vector<kernel::KernelTensor *> &outputs);
  bool LaunchBatches(const std::vector<kernel::KernelTensor *> &inputs,
                     const std::vector<kernel::KernelTensor *> &workspace,
                     const std::vector<kernel::KernelTensor *> &outputs);
  bool LaunchNoBatches(const std::vector<kernel::KernelTensor *> &inputs,
                       const std::vector<kernel::KernelTensor *> &workspace,
                       const std::vector<kernel::KernelTensor *> &outputs);
  template <typename T>
  bool LaunchMacBatches(const std::vector<kernel::KernelTensor *> &inputs,
                        const std::vector<kernel::KernelTensor *> &workspace,
                        const std::vector<kernel::KernelTensor *> &outputs);
  template <typename T>
  bool LaunchMacNoBatches(const std::vector<kernel::KernelTensor *> &inputs,
                          const std::vector<kernel::KernelTensor *> &workspace,
                          const std::vector<kernel::KernelTensor *> &outputs);

  std::string kernel_type_{"Unknown"};
  TypeId input_type_1_{kTypeUnknown};
  TypeId input_type_2_{kTypeUnknown};
  std::vector<size_t> input_shape_1_;
  std::vector<size_t> input_shape_2_;
  std::vector<size_t> output_shape_;
  size_t batches_{1};
  size_t in1dim_{1};
  size_t in2dim_{1};
  size_t outdim_{1};
  const size_t max_dims_{7};
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_GER_CPU_KERNEL_H_
