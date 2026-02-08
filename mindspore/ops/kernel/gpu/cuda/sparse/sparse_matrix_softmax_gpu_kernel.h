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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_SPARSE_MATRIX_SOFTMAX_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_SPARSE_MATRIX_SOFTMAX_GPU_KERNEL_H_

#include "kernel/gpu/cuda_impl/cuda_ops/sparse_matrix_softmax_impl.cuh"
#include <vector>
#include <utility>
#include <string>
#include <memory>
#include <map>
#include <algorithm>
#include <functional>
#include "kernel/gpu/cuda_impl/cuda_ops/cuda_common.h"
#include "kernel/gpu/cuda_impl/cuda_class/cuda_class_common.h"
#include "kernel/gpu/gpu_kernel.h"
#include "kernel/gpu/gpu_kernel_factory.h"

namespace mindspore {
namespace kernel {
class SparseMatrixSoftmaxGpuKernelMod : public NativeGpuKernelMod {
 public:
  SparseMatrixSoftmaxGpuKernelMod() = default;
  ~SparseMatrixSoftmaxGpuKernelMod() override = default;

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override {
    cuda_stream_ = stream_ptr;
    return kernel_func_(this, inputs, workspace, outputs);
  }

  std::vector<KernelAttr> GetOpSupport() override;

 protected:
  void InitSizeLists() {
    // output size list
    output_size_list_.push_back(dense_shape_elements_ * index_unit_size_);
    output_size_list_.push_back(batch_pointers_elements_ * index_unit_size_);
    output_size_list_.push_back(row_pointers_elements_ * index_unit_size_);
    output_size_list_.push_back(col_indices_elements_ * index_unit_size_);
    output_size_list_.push_back(values_elements_ * data_unit_size_);
  }

 private:
  template <typename DataType, typename IndexType>
  bool LaunchKernel(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                    const std::vector<KernelTensor *> &outputs);

  using SparseMatrixSoftmaxLaunchFunc =
    std::function<bool(SparseMatrixSoftmaxGpuKernelMod *, const std::vector<kernel::KernelTensor *> &,
                       const std::vector<kernel::KernelTensor *> &, const std::vector<kernel::KernelTensor *> &)>;
  static std::vector<std::pair<KernelAttr, SparseMatrixSoftmaxLaunchFunc>> func_list_;
  SparseMatrixSoftmaxLaunchFunc kernel_func_;

 private:
  void *cuda_stream_{nullptr};
  size_t data_unit_size_{1};
  size_t index_unit_size_{1};

  std::vector<size_t> input_dense_shape_;
  std::vector<size_t> input_batch_pointers_;
  std::vector<size_t> input_row_pointers_;
  std::vector<size_t> input_col_indices_;
  std::vector<size_t> input_values_;
  size_t dense_shape_elements_{0};
  size_t batch_pointers_elements_{0};
  size_t row_pointers_elements_{0};
  size_t col_indices_elements_{0};
  size_t values_elements_{0};
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_SPARSE_MATRIX_SOFTMAX_GPU_KERNEL_H_
