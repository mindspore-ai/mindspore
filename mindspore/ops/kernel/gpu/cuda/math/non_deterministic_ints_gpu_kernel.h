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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_NON_DETERMINISTIC_INTS_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_NON_DETERMINISTIC_INTS_GPU_KERNEL_H_

#include <vector>
#include <memory>
#include <functional>
#include <map>
#include <utility>
#include "mindspore/ops/infer/non_deterministic_ints.h"
#include "kernel/gpu/gpu_kernel.h"
#include "include/runtime/hardware_abstract/kernel_base/ms_factory.h"
namespace mindspore {
namespace kernel {
class NonDeterministicIntsGpuKernelMod : public NativeGpuKernelMod {
 public:
  NonDeterministicIntsGpuKernelMod() = default;
  ~NonDeterministicIntsGpuKernelMod() override = default;

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *cuda_stream) override;

 protected:
  std::vector<KernelAttr> GetOpSupport() override;

 protected:
  void ResetResource() noexcept {
    input_num_ = 1;
    output_num_ = 1;
    output_size_list_.clear();
    workspace_size_list_.clear();
  }

 private:
  template <typename T>
  bool LaunchKernel(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                    const std::vector<KernelTensor *> &outputs, void *cuda_stream);
  using KernelFunc = std::function<bool(NonDeterministicIntsGpuKernelMod *, const std::vector<kernel::KernelTensor *> &,
                                        const std::vector<kernel::KernelTensor *> &,
                                        const std::vector<kernel::KernelTensor *> &, void *)>;
  KernelFunc kernel_func_{};
  static std::vector<std::pair<KernelAttr, KernelFunc>> func_list_;

  int unit_input_size_;
  int unit_output_size_;
  int64_t input_num_{0};
  int64_t output_num_{0};
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_NON_DETERMINISTIC_INTS_GPU_KERNEL_H_
