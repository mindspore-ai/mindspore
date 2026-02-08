/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_TRACEV2_GRAD_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_TRACEV2_GRAD_GPU_KERNEL_H_
#include <vector>
#include <string>
#include <memory>
#include <utility>
#include <algorithm>
#include <functional>
#include <map>
#include "include/runtime/hardware_abstract/kernel_base/ms_factory.h"
#include "kernel/gpu/gpu_kernel.h"
#include "kernel/gpu/gpu_kernel_factory.h"
#include "kernel/gpu/cuda_impl/cuda_ops/tracev2_grad_impl.cuh"
#include "kernel/gpu/cuda_impl/cuda_ops/transpose_impl.cuh"

namespace mindspore {
namespace kernel {
class TraceV2GradGpuKernelMod : public NativeGpuKernelMod {
 public:
  TraceV2GradGpuKernelMod() { ResetResource(); }
  ~TraceV2GradGpuKernelMod() override = default;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *cuda_stream) override {
    cuda_stream_ = cuda_stream;
    return kernel_func_(this, inputs, workspace, outputs);
  }

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

 protected:
  void ResetResource() noexcept {
    offset_ = 1;
    axis1_ = 0;
    axis2_ = 1;
    din_size_ = 0;
    mat_size_ = 0;
    min_size_ = 0;
    mat_row_size_ = 0;
    data_unit_size_ = 0;
    output_size_list_.clear();
    workspace_size_list_.clear();
  }

  std::vector<KernelAttr> GetOpSupport() override;

 private:
  template <typename T>
  bool LaunchKernel(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                    const std::vector<KernelTensor *> &outputs);

  using TraceV2GradFunc =
    std::function<bool(TraceV2GradGpuKernelMod *, const std::vector<kernel::KernelTensor *> &,
                       const std::vector<kernel::KernelTensor *> &, const std::vector<kernel::KernelTensor *> &)>;

 private:
  int64_t offset_;
  int64_t axis1_;
  int64_t axis2_;
  int64_t din_size_;
  int64_t mat_size_;
  int64_t min_size_;
  int64_t mat_row_size_;
  size_t data_unit_size_;
  std::vector<int64_t> input_shape_;
  std::vector<int32_t> input_perm_;
  TraceV2GradFunc kernel_func_{};
  void *cuda_stream_{nullptr};
  static std::vector<std::pair<KernelAttr, TraceV2GradFunc>> func_list_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_TRACEV2_GPU_KERNEL_H_
