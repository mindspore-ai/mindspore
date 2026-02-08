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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_CONJUGATE_TRANSPOSE_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_CONJUGATE_TRANSPOSE_H_
#include <vector>
#include <string>
#include <utility>
#include <map>
#include <complex>
#include "kernel/gpu/gpu_kernel.h"
#include "include/runtime/hardware_abstract/kernel_base/ms_factory.h"

namespace mindspore {
namespace kernel {
class ConjugateTransposeGpuKernelMod : public NativeGpuKernelMod {
 public:
  ConjugateTransposeGpuKernelMod() { ResetResource(); }
  ~ConjugateTransposeGpuKernelMod() override = default;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *cuda_stream) override {
    if (is_null_input_) {
      return true;
    }
    cuda_stream_ = cuda_stream;
    return kernel_func_(this, inputs, workspace, outputs);
  }

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  std::vector<KernelAttr> GetOpSupport() override;

 private:
  void ResetResource() noexcept;

  template <typename T, typename S>
  bool LaunchKernel(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                    const std::vector<KernelTensor *> &outputs);
  template <typename T, typename S>
  bool LaunchComplexKernel(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                           const std::vector<KernelTensor *> &outputs);
  using ConjugateTransposeFunc =
    std::function<bool(ConjugateTransposeGpuKernelMod *, const std::vector<kernel::KernelTensor *> &,
                       const std::vector<kernel::KernelTensor *> &, const std::vector<kernel::KernelTensor *> &)>;

 private:
  size_t unit_size_one_{1};
  size_t unit_size_two_{1};
  size_t out_unit_size_{1};
  bool is_null_input_{false};
  bool states_init_{false};
  size_t shape_size_{};
  size_t x_one_count_{};
  size_t x_two_count_{};
  size_t y_count_{};

  void *cuda_stream_{nullptr};
  size_t input_stride[7];
  size_t output_stride[7];
  std::vector<size_t> x_one_shape_;
  std::vector<size_t> x_two_shape_;
  std::vector<size_t> y_shape_;
  ConjugateTransposeFunc kernel_func_{};
  static std::vector<std::pair<KernelAttr, ConjugateTransposeFunc>> func_list_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_CONJUGATE_TRANSPOSE_H_
