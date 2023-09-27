/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_MASKED_SELECTED_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_MASKED_SELECTED_CPU_KERNEL_H_

#include <vector>
#include <utility>
#include <map>

#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"

namespace mindspore {
namespace kernel {
class MaskedSelectCpuKernelMod : public NativeCpuKernelMod {
 public:
  MaskedSelectCpuKernelMod() = default;
  ~MaskedSelectCpuKernelMod() override = default;

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &,
              const std::vector<KernelTensor *> &outputs) override {
    return kernel_func_(this, inputs, outputs);
  }

  std::vector<KernelAttr> GetOpSupport() override;

 protected:
  void SyncOutputShape() override;

 private:
  template <typename T>
  bool LaunchKernel(const std::vector<kernel::KernelTensor *> &inputs,
                    const std::vector<kernel::KernelTensor *> &outputs);
  using MaskedSelectFunc = std::function<bool(MaskedSelectCpuKernelMod *, const std::vector<kernel::KernelTensor *> &,
                                              const std::vector<kernel::KernelTensor *> &)>;
  static std::vector<std::pair<KernelAttr, MaskedSelectFunc>> func_list_;
  MaskedSelectFunc kernel_func_;
  void ResetResource() noexcept;
  std::vector<int64_t> input_shape_a_;
  std::vector<int64_t> input_shape_b_;
  std::vector<int64_t> output_shape_;
  size_t tensor_size_;
  size_t real_output_size_;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_MASKED_SELECTED_CPU_KERNEL_H_
