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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_MATRIX_SET_DIAG_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_MATRIX_SET_DIAG_KERNEL_H_

#include <vector>
#include <memory>
#include <string>
#include <algorithm>
#include <utility>
#include <map>
#include "kernel/gpu/gpu_kernel.h"
#include "kernel/gpu/gpu_kernel_factory.h"

namespace mindspore {
namespace kernel {
class MatrixSetDiagV3GpuKernelMod : public NativeGpuKernelMod {
 public:
  MatrixSetDiagV3GpuKernelMod() = default;
  ~MatrixSetDiagV3GpuKernelMod() override = default;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &,
              const std::vector<KernelTensor *> &outputs, void *cuda_stream) override {
    if (is_null_input_) {
      return true;
    }
    cuda_stream_ = cuda_stream;
    return kernel_func_(this, inputs, outputs);
  }

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  std::vector<KernelAttr> GetOpSupport() override;

 private:
  template <typename T>
  bool LaunchKernel(const std::vector<kernel::KernelTensor *> &inputs,
                    const std::vector<kernel::KernelTensor *> &outputs);
  using MatrixDiagV3Func =
    std::function<bool(MatrixSetDiagV3GpuKernelMod *, const std::vector<kernel::KernelTensor *> &,
                       const std::vector<kernel::KernelTensor *> &)>;

  void ResetResource() noexcept;

  bool is_single_diag_{true};
  bool is_null_input_{false};
  int lower_{0};
  int upper_{0};
  int inner_rows_{0};
  int inner_cols_{0};
  int num_diags_{0};
  int expected_num_diags_{0};
  int max_diag_len_{0};
  int outer_batch_{1};
  size_t diagonal_count_{1};
  size_t k_count_{1};
  void *cuda_stream_{nullptr};
  // <super_matrix_diag_align, sub_matrix_diag_align>
  std::pair<MatrixDiag::Alignment, MatrixDiag::Alignment> alignment_{MatrixDiag::RIGHT, MatrixDiag::LEFT};
  static std::vector<std::pair<KernelAttr, MatrixDiagV3Func>> func_list_;
  MatrixDiagV3Func kernel_func_;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_MATRIX_SET_DIAG_KERNEL_H_
