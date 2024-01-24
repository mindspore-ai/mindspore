/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_CORRELATE_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_CORRELATE_CPU_KERNEL_H_

#include <vector>
#include <complex>
#include <utility>
#include <string>
#include <map>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"

namespace mindspore {
namespace kernel {
class CorrelateCpuKernelMod : public NativeCpuKernelMod {
 public:
  CorrelateCpuKernelMod() = default;
  ~CorrelateCpuKernelMod() override = default;

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;
  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;
  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs) override {
    return kernel_func_(this, inputs, outputs);
  }
  template <typename T>
  void CorrelatePad(T *source_array, T *paded_array, int64_t padded_array_size);

 protected:
  std::vector<KernelAttr> GetOpSupport() override;

 private:
  template <typename T_in, typename T_out>
  bool LaunchKernel(const std::vector<kernel::KernelTensor *> &inputs,
                    const std::vector<kernel::KernelTensor *> &outputs);
  template <typename T>
  bool LaunchComplexKernel(const std::vector<kernel::KernelTensor *> &inputs,
                           const std::vector<kernel::KernelTensor *> &outputs);
  using CorrelateFunc = std::function<bool(CorrelateCpuKernelMod *, const std::vector<KernelTensor *> &,
                                           const std::vector<KernelTensor *> &)>;
  static std::vector<std::pair<KernelAttr, CorrelateFunc>> func_list_;
  CorrelateFunc kernel_func_;

  int64_t a_size_;
  int64_t v_size_;
  bool a_ge_v_;
  int64_t long_size_;
  int64_t short_size_;
  mindspore::PadMode mode_type_;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_CORRELATE_CPU_KERNEL_H_
