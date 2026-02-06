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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_SPARSE_CSR_SPARSE_MATRIX_TO_DENSE_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_SPARSE_CSR_SPARSE_MATRIX_TO_DENSE_GPU_KERNEL_H_

#include <cuda_runtime_api.h>
#include <vector>
#include <map>
#include <utility>
#include "include/utils/anfalgo.h"
#include "kernel/gpu/gpu_kernel.h"
#include "kernel/gpu/gpu_kernel_factory.h"
#include "kernel/gpu/cuda_impl/cuda_ops/csr_sparse_matrix_to_dense.cuh"

namespace mindspore {
namespace kernel {
class CSRSparseMatrixToDenseGpuKernelMod : public NativeGpuKernelMod {
 public:
  CSRSparseMatrixToDenseGpuKernelMod() = default;
  ~CSRSparseMatrixToDenseGpuKernelMod() override = default;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override {
    MS_EXCEPTION_IF_NULL(kernel_func_);
    return kernel_func_(this, inputs, workspace, outputs, stream_ptr);
  }

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  void ResetResource() noexcept;

 protected:
  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs);
  void InitSizeLists();
  template <typename T, typename S>
  bool LaunchKernel(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                    const std::vector<KernelTensor *> &outputs, void *stream_ptr);
  std::vector<KernelAttr> GetOpSupport() override;

 private:
  using CSRSparseMatrixToDenseFunc =
    std::function<bool(CSRSparseMatrixToDenseGpuKernelMod *, const std::vector<KernelTensor *> &,
                       const std::vector<KernelTensor *> &, const std::vector<KernelTensor *> &, void *)>;
  static std::vector<std::pair<KernelAttr, CSRSparseMatrixToDenseFunc>> func_list_;
  CSRSparseMatrixToDenseFunc kernel_func_;
  bool is_null_input_{false};
  size_t dense_shape_size_{0};
  size_t batch_ptr_size_{0};
  size_t row_ptr_size_{0};
  size_t col_indices_size_{0};
  size_t values_size_{0};
  size_t output_size_{0};
  size_t ndim{0};
  size_t rows{0};
  size_t nums{0};
  std::vector<int64_t> dense_shape_shape_;
  std::vector<int64_t> batch_ptr_shape_;
  std::vector<int64_t> row_ptr_shape_;
  std::vector<int64_t> col_indices_shape_;
  std::vector<int64_t> values_shape_;
  std::vector<int64_t> output_shape_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_SPARSE_CSR_SPARSE_MATRIX_TO_DENSE_GPU_KERNEL_H_
