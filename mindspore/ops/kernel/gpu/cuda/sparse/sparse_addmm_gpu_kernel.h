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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_SPARSE_ADDMM_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_SPARSE_ADDMM_GPU_KERNEL_H_
#include <vector>
#include <string>
#include <memory>
#include <utility>
#include <algorithm>
#include <functional>
#include <map>
#include "mindspore/ops/infer/sparse_addmm.h"
#include "include/runtime/hardware_abstract/kernel_base/ms_factory.h"
#include "kernel/gpu/gpu_kernel.h"
#include "kernel/gpu/gpu_kernel_factory.h"
#include "kernel/gpu/cuda_impl/cuda_ops/sparse_addmm_impl.cuh"

namespace mindspore {
namespace kernel {
class SparseAddmmGpuKernelMod : public NativeGpuKernelMod {
 public:
  SparseAddmmGpuKernelMod() { ResetResource(); }
  ~SparseAddmmGpuKernelMod() override = default;

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

 protected:
  void ResetResource() noexcept {
    input_values_num_ = 0;
    mat1_values_num_ = 0;
    y_values_num_ = 0;
    mat1_col_ = 0;
    mat2_col_ = 0;
    is_null_input_ = false;
    workspace_size_list_.clear();
    output_size_list_.clear();
  }

 private:
  template <typename T, typename S>
  bool LaunchKernel(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                    const std::vector<KernelTensor *> &outputs);
  using SparseAddmmFunc =
    std::function<bool(SparseAddmmGpuKernelMod *, const std::vector<kernel::KernelTensor *> &,
                       const std::vector<kernel::KernelTensor *> &, const std::vector<kernel::KernelTensor *> &)>;

 private:
  int64_t unit_indices_size_{1};
  int64_t unit_values_size_{1};
  int64_t input_values_num_{};
  int64_t output_values_num_{};
  int64_t mat1_values_num_{};
  int64_t y_values_num_{};
  int64_t mat1_row_{};
  int64_t mat2_row_{};
  int64_t mat3_row_{};
  int64_t mat1_col_{};
  int64_t mat2_col_{};
  int64_t mat3_col_{};
  int64_t output_row_{};
  int64_t output_col_{};
  SparseAddmmFunc kernel_func_{};
  bool is_null_input_{false};
  void *cuda_stream_{nullptr};
  static std::vector<std::pair<KernelAttr, SparseAddmmFunc>> func_list_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_SPARSE_ADDMM_GPU_KERNEL_H_
