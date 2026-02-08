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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_SPARSE_CSR_SPARSE_MATRIX_TO_SPARSE_TENSOR_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_SPARSE_CSR_SPARSE_MATRIX_TO_SPARSE_TENSOR_GPU_KERNEL_H_

#include <cuda_runtime_api.h>
#include <cusparse.h>
#include <map>
#include <memory>
#include <vector>
#include <utility>
#include "mindspore/ops/infer/csr_sparse_matrix_to_sparse_tensor.h"
#include "plugin/gpu/res_manager/symbol_interface/cuda_driver.h"
#include "kernel/gpu/cuda_impl/cuda_ops/csr_sparse_matrix_to_sparse_tensor_gpu_kernel.cuh"
#include "kernel/gpu/gpu_kernel.h"
#include "kernel/gpu/gpu_kernel_factory.h"

constexpr size_t kCSRSparseMatrixToSparseTensorInputsNum = 5;
constexpr size_t kCSRSparseMatrixToSparseTensorOutputsNum = 3;
constexpr size_t kBatchCSR = 3;

namespace mindspore {
namespace kernel {
class CSRSparseMatrixToSparseTensorGpuKernelMod : public NativeGpuKernelMod {
 public:
  CSRSparseMatrixToSparseTensorGpuKernelMod() {
    ResetResource();
    handle_ = device::gpu::GPUDeviceManager::GetInstance().GetCuSparseHandle();
  }
  ~CSRSparseMatrixToSparseTensorGpuKernelMod() override = default;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override {
    MS_EXCEPTION_IF_NULL(kernel_func_);
    return kernel_func_(this, inputs, workspace, outputs, stream_ptr);
  }

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;
  void ResetResource() noexcept;

 protected:
  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;
  void InitSizeLists();
  template <typename T, typename S>
  bool LaunchKernel(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                    const std::vector<KernelTensor *> &outputs, void *stream_ptr);
  std::vector<KernelAttr> GetOpSupport() override;

 private:
  using CSRSparseMatrixToSparseTensorFunc =
    std::function<bool(CSRSparseMatrixToSparseTensorGpuKernelMod *, const std::vector<KernelTensor *> &,
                       const std::vector<KernelTensor *> &, const std::vector<KernelTensor *> &, void *)>;
  static std::vector<std::pair<KernelAttr, CSRSparseMatrixToSparseTensorFunc>> func_list_;
  CSRSparseMatrixToSparseTensorFunc kernel_func_;
  cusparseHandle_t handle_{nullptr};
  bool is_null_input_{false};
  size_t input_dense_shape_size_{0};
  size_t input_batch_pointers_size_{0};
  size_t input_row_pointers_size_{0};
  size_t input_col_indices_size_{0};
  size_t input_values_size_{0};
  size_t output_indices_size_{0};
  size_t output_values_size_{0};
  size_t output_dense_shape_size_{0};
  std::vector<int64_t> input_dense_shape_shapes_;
  std::vector<int64_t> input_batch_pointers_shapes_;
  std::vector<int64_t> input_row_pointers_shapes_;
  std::vector<int64_t> input_col_indices_shapes_;
  std::vector<int64_t> input_values_shapes_;
  std::vector<int64_t> output_indices_shapes_;
  std::vector<int64_t> output_values_shapes_;
  std::vector<int64_t> output_dense_shape_shapes_;
  bool is_batch_csr_{false};
  int rank_{0};
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_SPARSE_CSR_SPARSE_M
