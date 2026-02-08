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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_APPLY_PROXIMAL_GRADIENT_DESCENT_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_APPLY_PROXIMAL_GRADIENT_DESCENT_KERNEL_H_

#include <vector>
#include <memory>
#include <functional>
#include <map>
#include <utility>
#include "mindspore/ops/infer/sparse_apply_proximal_gradient_descent.h"
#include "kernel/gpu/gpu_kernel.h"
#include "kernel/gpu/cuda_impl/cuda_ops/sparse_apply_proximal_gradient_descent_impl.cuh"
#include "include/runtime/hardware_abstract/kernel_base/ms_factory.h"
namespace mindspore {
namespace kernel {
class SparseApplyProximalGradientDescentGpuKernelMod : public NativeGpuKernelMod {
 public:
  SparseApplyProximalGradientDescentGpuKernelMod() = default;
  ~SparseApplyProximalGradientDescentGpuKernelMod() override = default;

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override {
    cuda_stream_ = stream_ptr;
    return kernel_func_(this, inputs, workspace, outputs);
  }

 protected:
  std::vector<KernelAttr> GetOpSupport() override;

 private:
  template <typename T, typename S>
  bool LaunchKernel(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                    const std::vector<KernelTensor *> &outputs);
  using KernelFunc =
    std::function<bool(SparseApplyProximalGradientDescentGpuKernelMod *, const std::vector<kernel::KernelTensor *> &,
                       const std::vector<kernel::KernelTensor *> &, const std::vector<kernel::KernelTensor *> &)>;
  KernelFunc kernel_func_{};
  static std::vector<std::pair<KernelAttr, KernelFunc>> func_list_;

  int unit_size_;
  int64_t unit_var_size_{1};
  int64_t unit_indices_size_{1};
  size_t input_elements_;
  size_t indices_size_;
  void *cuda_stream_{nullptr};
  int global_indices_shape_;
  size_t batch_size_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_SPARSE_APPLY_PROXIMAL_GRADIENT_DESCENT_KERNEL_H_
