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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_PARALLEL_CONCAT_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_PARALLEL_CONCAT_CPU_KERNEL_H_

#include <vector>
#include <utility>
#include <complex>
#include <map>
#include <functional>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"

namespace mindspore {
namespace kernel {
using complex64 = std::complex<float>;
using complex128 = std::complex<double>;

class ParallelConcatCpuKernelMod : public NativeCpuKernelMod {
 public:
  ParallelConcatCpuKernelMod() = default;
  ~ParallelConcatCpuKernelMod() override = default;

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs) override {
    return kernel_func_(this, inputs, outputs);
  }

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

 protected:
  void ResetResource() noexcept {
    input_num_ = 0;
    input_flat_shape_list_.clear();
  }
  std::vector<KernelAttr> GetOpSupport() override;

 private:
  template <typename T>
  bool LaunchKernel(const std::vector<kernel::KernelTensor *> &inputs,
                    const std::vector<kernel::KernelTensor *> &outputs);
  using PCFunc = std::function<bool(ParallelConcatCpuKernelMod *, const std::vector<kernel::KernelTensor *> &,
                                    const std::vector<kernel::KernelTensor *> &)>;

 private:
  size_t input_num_;
  PCFunc kernel_func_;
  bool is_null_input_{false};
  std::vector<ShapeVector> input_flat_shape_list_;
  static std::vector<std::pair<KernelAttr, PCFunc>> func_list_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_PARALLEL_CONCAT_CPU_KERNEL_H_
