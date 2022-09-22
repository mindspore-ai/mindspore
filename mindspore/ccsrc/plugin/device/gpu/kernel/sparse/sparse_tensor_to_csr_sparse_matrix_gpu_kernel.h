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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_SPARSE_SPARSE_TENSOR_TO_CSR_SPARSE_MATRIX_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_SPARSE_SPARSE_TENSOR_TO_CSR_SPARSE_MATRIX_GPU_KERNEL_H_

#include <vector>
#include <utility>
#include <string>
#include <map>
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/kernel_constants.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/sparse_tensor_to_csr_sparse_matrix_impl.cuh"

namespace mindspore {
namespace kernel {
class SparseTensorToCSRSparseMatrixGpuKernelMod : public NativeGpuKernelMod {
 public:
  SparseTensorToCSRSparseMatrixGpuKernelMod() = default;
  ~SparseTensorToCSRSparseMatrixGpuKernelMod() override = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *cuda_stream) override {
    stream = reinterpret_cast<cudaStream_t>(cuda_stream);
    return kernel_func_(this, inputs, workspace, outputs);
  }

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs, const std::map<uint32_t, tensor::TensorPtr> &) override;

 protected:
  std::vector<KernelAttr> GetOpSupport() override;

 private:
  template <typename IndiceType, typename DataType>
  bool LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                    const std::vector<AddressPtr> &outputs);
  using SparseTensorToCSRSparseMatrixFunc =
    std::function<bool(SparseTensorToCSRSparseMatrixGpuKernelMod *, const std::vector<kernel::AddressPtr> &,
                       const std::vector<kernel::AddressPtr> &, const std::vector<kernel::AddressPtr> &)>;

  static std::vector<std::pair<KernelAttr, SparseTensorToCSRSparseMatrixFunc>> func_list_;

 private:
  float alpha_;
  size_t unit_size_{1};
  size_t input_elements_{};
  void *cuda_stream_{nullptr};
  int elements[3] = {0, 0, 0};
  cudaStream_t stream;
  cusparseHandle_t handle_{nullptr};
  int row_num;
  int prev_batch;
  int batch_size;
  int cur_batch;
  int total_nnz;
  int rank_;
  int temp_nnz;
  std::vector<int> x_indices_ptr_test;
  std::vector<int> y_batch_pointers_ptr_test;
  std::vector<int> batch_size_ptr;
  std::vector<int> x_dense_shape_ptr_test;
  SparseTensorToCSRSparseMatrixFunc kernel_func_{};
  template <typename _Tp>
  class complex;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_SPARSE_SPARSE_TENSOR_TO_CSR_SPARSE_MATRIX_GPU_KERNEL_H_
